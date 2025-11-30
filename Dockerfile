FROM mambaorg/micromamba:latest

USER root

# Install git and other dependencies
RUN apt-get update
RUN apt-get install -y git nano curl wget && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN export UV_PROJECT_ENVIRONMENT=/opt/conda/envs/llm-foundry
ENV UV_PROJECT_ENVIRONMENT=/opt/conda/envs/llm-foundry
ENV CONDA_DEFAULT_ENV=llm-foundry
ENV PATH=/opt/conda/envs/llm-foundry/bin:$PATH

# Clone llm-foundry repo and set up environment
RUN git clone -b tokenize-datasets-process-datasets https://github.com/LocalResearchGroup/llm-foundry.git /llm-foundry

WORKDIR /llm-foundry
run git status

RUN micromamba create -n llm-foundry python=3.12 uv cuda -c nvidia/label/12.4.1 -c conda-forge
RUN micromamba shell init -s bash
RUN . ~/.bashrc
RUN micromamba activate llm-foundry && \
    uv python pin 3.12 && \
    uv sync --dev --extra gpu && \
    uv pip install --upgrade huggingface_hub && \
    uv pip install --upgrade datasets && \
    uv sync --dev && \
    uv sync --dev --extra gpu --extra flash --no-cache


# Initialize conda in bash and activate environment by default
RUN echo "eval \"\$(micromamba shell hook --shell bash)\"" >> ~/.bashrc && \
    echo "micromamba activate llm-foundry" >> ~/.bashrc

RUN cat ~/.bashrc

# Open port to view Aim dashboard live from the container (optional) - Not related to aim remote upload server.
EXPOSE 43800

# Default shell with environment activated
CMD ["/bin/bash"]
