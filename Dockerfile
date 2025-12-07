FROM mambaorg/micromamba:latest

run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo "################################################################"
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# run echo -=-==-=-=-=-=-===-=-=-=-=
# ENV UV_PROJECT_ENVIRONMENT=/opt/conda/envs/llm-foundry
# ENV CONDA_DEFAULT_ENV=llm-foundry
# ENV PATH=/opt/conda/envs/llm-foundry/bin:$PATH
# ENV UV_PROJECT_ENVIRONMENT=/opt/conda/envs/llm-foundry
# USER root
#
# # Install git and other dependencies
# RUN apt-get update && apt-get install -y git nano curl wget && apt-get clean && rm -rf /var/lib/apt/lists/*
#
# # Clone llm-foundry repo and set up environment
# RUN git clone -b tokenize-datasets-process-datasets https://github.com/LocalResearchGroup/llm-foundry.git /llm-foundry
#
# WORKDIR /llm-foundry
# run git status && ls -la
# run echo "###########################################"
# run ls /
# run echo "###########################################"
#
# run micromamba create -n llm-foundry python=3.12 uv cuda -c nvidia/label/12.4.1 -c conda-forge && \
#     micromamba run -n llm-foundry uv python pin 3.12 && \
#     micromamba run -n llm-foundry uv sync --dev --extra gpu && \
#     micromamba run -n llm-foundry uv sync --dev --extra gpu --extra flash --no-cache
#
#
#
# # Initialize conda in bash and activate environment by default
# RUN echo "eval \"\$(micromamba shell hook --shell bash)\"" >> ~/.bashrc && \
#     echo "micromamba activate llm-foundry" >> ~/.bashrc
#
# # Open port to view Aim dashboard live from the container (optional) - Not related to aim remote upload server.
# EXPOSE 43800
#
# # Default shell with environment activated
# CMD ["/bin/bash"]
#
# # FROM mambaorg/micromamba:latest
# #
# # USER root
# #
# # # Install git and other dependencies
# # RUN apt-get update && apt-get install -y git nano curl wget tree && apt-get clean && rm -rf /var/lib/apt/lists/*
# #
# # # Clone llm-foundry repo and set up environment
# # RUN git clone -b tokenize-datasets-process-datasets hhttps://github.com/LocalResearchGroup/llm-foundry.git -b tokenize-datasets /llm-foundry && \
# #     cd /llm-foundry && \
# #     micromamba create -n llm-foundry python=3.12 uv cuda -c nvidia/label/12.4.1 -c conda-forge && \
# #     export UV_PROJECT_ENVIRONMENT=/opt/conda/envs/llm-foundry && \
# #     micromamba run -n llm-foundry uv python pin 3.12 && \
# #     micromamba run -n llm-foundry uv sync --dev --extra gpu && \
# #     micromamba run -n llm-foundry uv sync --dev --extra gpu --extra flash --no-cache
# #
# # ENV UV_PROJECT_ENVIRONMENT=/opt/conda/envs/llm-foundry
# # ENV CONDA_DEFAULT_ENV=llm-foundry
# # ENV PATH=/opt/conda/envs/llm-foundry/bin:$PATH
# #
# # WORKDIR /llm-foundry
# #
# # run tree -d
# #
# # # Initialize conda in bash and activate environment by default
# # RUN echo "eval \"\$(micromamba shell hook --shell bash)\"" >> ~/.bashrc && \
# #     echo "micromamba activate llm-foundry" >> ~/.bashrc
# #
# # # Open port to view Aim dashboard live from the container (optional) - Not related to aim remote upload server.
# # EXPOSE 43800
# #
# # # Default shell with environment activated
# # CMD ["/bin/bash"]
# #
