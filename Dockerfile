FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Remove torchvision from base image — incompatible with torch 2.8 and unused
RUN pip uninstall -y torchvision

RUN pip install --no-cache-dir \
    whisperx \
    pyannote.audio \
    speechbrain \
    numpy \
    "transformers<4.50"

WORKDIR /app

ENTRYPOINT ["python", "tools/whisperx_diarize.py"]
