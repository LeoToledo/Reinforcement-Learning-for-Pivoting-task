Reinforcement Learning for Pivoting Task

Para rodar, é necessário ter mujoco, mujoco_py e gym(via pip) instalados.

Após clonar o arquivo, rode os seguintes comandos:

      cd pivoting-env/
      pip install -e .

Para rodar o ppo pelo terminal:

    python -m spinup.run ppo --env pivoting-v0 --exp_name Sim --epochs 500 --max_ep_len 4000 --steps_per_epoch 8000 --hid[h] [32,24,10]
