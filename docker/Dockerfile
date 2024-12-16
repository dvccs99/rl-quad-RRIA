FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04


# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
    
RUN apt-get update

# Set environment variable to avoid dialog prompts
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git

RUN apt update && apt install -y python3-pip build-essential

RUN git clone https://github.com/dvccs99/rl-quad-RRIA.git

RUN pip install poetry
