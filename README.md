# Using Deep Reinforcement Learning to Achieve Locomotion for a Simulated Quadruped Robot

Repositório para a monografia de conclusão de curso do residente Daniel Salvador
da terceira turma da residência de robótica e I.A do CIn/UFPE.
Para rodar o experimento, instale o ambiente poetry:

```bash
poetry install
```

e rode o arquivo `sac_continuous_action.py`. Para alterar os parâmetros do algoritmo, acesse esse mesmo arquivo e altere-os. Para alterar os parâmetros das funções de recompensa ou mudar as funções em si, acesse `rl_quad/envs/quad_env.py` que é o arquivo que contém o ambiente Gymnasium utilizado.

A pasta `Docker` contém os arquivos docker para rodar o ambiente no cluster (PC_GAMER) da residência. 

O arquivo `replay_buffer.py` contém o replay buffer customizado para uso pelo SAC.

O arquivo `simulate_environment.py` permite visualizar o robô e o ambiente em que ele se encontra.

Github: https://github.com/dvccs99/rl-quad-RRIA

WandB: https://wandb.ai/dvccs-universidade-federal-de-pernambuco/Quad_Mujoco
