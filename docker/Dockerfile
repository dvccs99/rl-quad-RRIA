FROM pytorch/pytorch


RUN apt-get update && apt-get install -y git
RUN apt-get install nano -y
RUN apt-get install xvfb -y
RUN export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
RUN git clone https://github.com/dvccs99/rl-quad-RRIA.git -b development

RUN pip install "gymnasium[mujoco]"==0.29.1 tensorboard==2.18.0 wandb==0.13.11 \
git+https://github.com/DLR-RM/stable-baselines3 tyro==0.9.5 scipy==1.15 tyro moviepy==2.1.1
