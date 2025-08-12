# models/logic_python/1/model.py
import triton_python_backend_utils as pb_utils
import numpy as np
import io
import json
import logging
from PIL import Image
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("logic_python")

class TritonPythonModel:
    def initialize(self, args):
        self.model_ocr = "model_ocr"
        self.model_ocr_output_name = "OUTPUT"
        self.final_output_name = "FINAL_OUTPUT"
        self.prompt = (
            "Bạn là một hệ thống OCR thông minh. Hãy trích xuất tất cả thông tin có cấu trúc từ hình ảnh tài liệu.\n"
            "Trích xuất nội dung từ hình ảnh dưới dạng JSON hợp lệ với các trường sau:\n"
            "- Các thông tin metadata dạng key-value\n"
            "- Nếu có bảng trong ảnh, thêm trường 'tables' với cấu trúc:\n"
            "  - 'columns': danh sách tên cột\n"
            "  - 'rows': danh sách các dòng dữ liệu (tương ứng với tên cột)\n"
            "Yêu cầu:\n"
            "- Không mô tả, không markdown, không chú thích.\n"
            "- Chỉ trả về JSON thuần.\n"
            "- Giữ nguyên định dạng văn bản trong ảnh.\n"
            "- Nếu không có bảng thì không cần trường 'tables'.\n"
        )
        logger.info(f"logic_python initialized. Will call model: {self.model_ocr}")

    def _bytes_to_pil(self, b: bytes) -> Image.Image:
        if b.startswith(b"%PDF"):
            doc = fitz.open(stream=b, filetype="pdf")
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            img_bytes = pix.tobytes("png")
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return Image.open(io.BytesIO(b)).convert("RGB")

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                # Lấy bytes ảnh từ input "IMAGE"
                image_bytes = pb_utils.get_input_tensor_by_name(request, "IMAGE").as_numpy().tobytes()
                
                # Chuyển PDF/ảnh sang PNG bytes
                try:
                    pil_img = self._bytes_to_pil(image_bytes)
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    img_bytes_for_model = buf.getvalue()
                except Exception:
                    logger.warning("Fallback: dùng raw bytes cho model_ocr")
                    img_bytes_for_model = image_bytes

                # Tạo input cho model_ocr
                tensor_image = pb_utils.Tensor("IMAGE", np.frombuffer(img_bytes_for_model, dtype=np.uint8))
                tensor_prompt = pb_utils.Tensor("PROMPT", np.frombuffer(self.prompt.encode("utf-8"), dtype=np.uint8))

                # Gọi model_ocr qua BLS
                inference_request = pb_utils.InferenceRequest(
                    model_name=self.model_ocr,
                    requested_output_names=[self.model_ocr_output_name],
                    inputs=[tensor_image, tensor_prompt]
                )
                inference_response = inference_request.exec()

                if inference_response.has_error():
                    raise RuntimeError(inference_response.error().message())

                # Lấy output từ model_ocr
                out_tensor = pb_utils.get_output_tensor_by_name(inference_response, self.model_ocr_output_name)
                val0 = out_tensor.as_numpy()[0]
                if isinstance(val0, (bytes, bytearray)):
                    raw_output = val0.decode("utf-8", errors="ignore")
                else:
                    raw_output = str(val0)

                # Parse JSON nếu có thể
                try:
                    parsed = json.loads(raw_output)
                except:
                    parsed = {"_parse_error": True, "raw_output": raw_output}

                final_json_str = json.dumps(parsed, ensure_ascii=False)
                out_final_tensor = pb_utils.Tensor(self.final_output_name, np.array([final_json_str], dtype=object))
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_final_tensor]))

            except Exception as e:
                logger.exception("Error in execute()")
                err_obj = {"_error": str(e)}
                out_final_tensor = pb_utils.Tensor(self.final_output_name, np.array([json.dumps(err_obj, ensure_ascii=False)], dtype=object))
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_final_tensor]))

        return responses

    def finalize(self):
        logger.info("logic_python finalize called")
