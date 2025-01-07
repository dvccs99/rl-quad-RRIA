FROM pytorch/pytorch


RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/dvccs99/rl-quad-RRIA.git -b development

RUN pip install "gymnasium[mujoco]"==1.0.0 tensorboard==2.18.0 wandb==0.19.1 \
git+https://github.com/DLR-RM/stable-baselines3 tyro scipy tyro moviepy
