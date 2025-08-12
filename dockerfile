# FROM hieupth/tritonserverbuild:25.04
FROM hieupth/tritonserver:25.04-vllm
RUN pip install --no-cache-dir huggingface_hub transformers tokenizers numpy scikit-learn pyvi vllm autoawq torch demjson3 pymupdf flash-attention hf_xet


# Copy script bổ sung (nếu có)
# COPY ./scripts/ /workspace/scripts/
# WORKDIR /workspace

# Khi run container, Triton sẽ tự start bằng CMD
