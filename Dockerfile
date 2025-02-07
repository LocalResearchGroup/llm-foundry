FROM mambaorg/micromamba:latest

USER root

# Install git and other dependencies
RUN apt-get update && apt-get install -y git nano

# Clone llm-foundry repo and set up environment
RUN git clone https://github.com/LocalResearchGroup/llm-foundry.git /llm-foundry && \
    cd /llm-foundry && \
    micromamba create -n llm-foundry python=3.12 uv cuda -c nvidia/label/12.4.1 -c conda-forge && \
    export UV_PROJECT_ENVIRONMENT=/opt/conda/envs/llm-foundry && \
    micromamba run -n llm-foundry uv python pin 3.12 && \
    micromamba run -n llm-foundry uv sync --extra dev --extra gpu && \
    micromamba run -n llm-foundry uv sync --extra flash && \
    micromamba run -n llm-foundry uv pip install -e .

ENV UV_PROJECT_ENVIRONMENT=/opt/conda/envs/llm-foundry
ENV CONDA_DEFAULT_ENV=llm-foundry
ENV PATH=/opt/conda/envs/llm-foundry/bin:$PATH

WORKDIR /llm-foundry

# Initialize conda in bash and activate environment by default
RUN echo "eval \"\$(micromamba shell hook --shell bash)\"" >> ~/.bashrc && \
    echo "micromamba activate llm-foundry" >> ~/.bashrc

# Default shell with environment activated
CMD ["/bin/bash"]
