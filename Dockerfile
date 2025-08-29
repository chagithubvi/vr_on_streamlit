FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for building and audio support
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \ 
    gcc \
    ffmpeg \
    build-essential \
    python3-dev \
    libopus-dev \
    libvpx-dev \
    libavformat-dev \
    libavdevice-dev \
    libavfilter-dev \
    pkg-config \
    libxrender1 \
    libxext6 \
    libglib2.0-0 \
    libsm6 \
    libavcodec-dev \
    libswresample-dev \
    libavutil-dev \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
   

# Copy only requirements first for better cache
COPY requirements.txt /app/

# Upgrade pip, setuptools, wheel before installing requirements
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy rest of the app files
COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "main_cloud.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.enableCORS=false"]
