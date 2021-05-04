Reinforcement Learning for Pivoting Task

Para rodar, é necessário ter mujoco, mujoco_py e gym(via pip) instalados.

Após clonar o arquivo, rode os seguintes comandos:

      cd pivoting-env/
      pip install -e .

Para rodar: 

      python train_model.py
      python test_model.py


A modificação do tamanho do espaço de ações foi feita no arquivo ppo.py, na biblioteca spinup.
Para reproduzir a modificação, vá na função store da classe PPOBuffer(linha 29) e cole o seguinte trecho logo após o 
comando "self.obs_buf[self.ptr] = obs"(linha 34):

        ###############################################################################################
                                            #REFORMULANDO ESPAÇO DE AÇÕES
        act = np.array([act[0][5], act[0][7]])
        ###############################################################################################

Depois disso, vá na função ppo(linha 93) e adicione o seguinte código após o comando "act_dim = env.action_space.shape"
(linha 182 ou um pouco menos)

    ###############################################################################################
                                    # REFORMULANDO ESPAÇO DE AÇÕES
    act_dim = (2,)
    ###############################################################################################