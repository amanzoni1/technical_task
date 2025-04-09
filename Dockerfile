FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

WORKDIR /app

# Install Python and other dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install the requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install specific diffusers version
RUN pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git@723dbdd36300cd5a14000b828aaef87ba7e1fa68

# Copy the application code
COPY . .

# Set environment variables for Hugging Face
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

# Create entrypoint script to handle Hugging Face login
RUN echo '#!/bin/bash \n\
    if [ -n "$HF_TOKEN" ]; then \n\
    echo "Logging into Hugging Face..." \n\
    huggingface-cli login --token $HF_TOKEN \n\
    fi \n\
    exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Expose the port
EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]

# Set the command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
