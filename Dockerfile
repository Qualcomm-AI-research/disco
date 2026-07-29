FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# System deps for OpenCV / image libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir \
    diffusers==0.33.1 \
    transformers==4.40.0 \
    accelerate==1.4.0 \
    peft==0.10.0 \
    tokenizers==0.19.1 \
    numpy==1.26.4 \
    pillow==10.4.0 \
    tqdm==4.67.1 \
    "gradio>=6.14.0,<6.15.0" \
    "httptools>=0.7.1,<0.8.0" \
    "uvloop>=0.22.1,<0.23.0" \
    xformers==0.0.29.post1 \
    bitsandbytes==0.45.3 \
    einops==0.8.1 \
    protobuf \
    sentencepiece

COPY . .

RUN chmod -R 755 /app

