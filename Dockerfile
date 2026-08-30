FROM python:3.10-slim

# Install system dependencies including FFmpeg for audio processing and faster-whisper
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces requires a user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_THEME=soft \
    SYSTEM=spaces

WORKDIR $HOME/app

# Copy requirements and install dependencies
COPY --chown=user requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all application files
COPY --chown=user . $HOME/app

# Expose port 7860
EXPOSE 7860

# Run the web application
CMD ["python", "app.py"]
