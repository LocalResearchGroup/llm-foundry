FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks
RUN ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Set CUDA_HOME
ENV CUDA_HOME=/usr/local/cuda

# Copy the requirements.txt file into the Docker image
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies including flash-attention
RUN pip install uv --no-cache-dir
RUN uv venv --python 3.11
RUN uv pip install -r /tmp/requirements.txt
RUN uv pip install flash-attn==2.6.3 --no-build-isolation

# Clone and install llm-foundry
RUN git clone https://github.com/LocalResearchGroup/llm-foundry.git && \
    cd llm-foundry && \
    uv pip install -e ".[gpu]"