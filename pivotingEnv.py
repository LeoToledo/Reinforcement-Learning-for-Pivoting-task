import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'pivoting.xml', 2)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        
    #Modelando a reward
        #Enquanto não cair, mas não completar o objetivo, a reward é -1
        if(ob[4] > 0.07): 
            reward = -1
        #Quando cair, a reward é -5
        elif(ob[4] <= 0.07):
            reward = -5
        #Caso atinja o ângulo desejado, a reward deve ser modelada no código principal
        #Modelando se done é True ou não. Quando o objeto cair, done==1. O caso de limite de tempo deve ser modelado no código principal
        done = 0
        if(reward == -5):
            done = 1
        
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        obs = np.concatenate([self.sim.data.qpos]).ravel()
        #Vamos converter o angulo da ferramenta para graus
        conv_rad = np.arctan2(2*(obs[5]*obs[8] + obs[6]*obs[7]), (1 - 2*(obs[7]**2 + obs[8]**2)))
        obs_degrees = 180*conv_rad/np.pi
        obs[5] = obs_degrees
        #Convertendo o Hinge do Arm para Graus
        obs[0] = 180*obs[0]/np.pi
        return obs

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent + 0.5


###########################
#obs[0] - Hinge - Angulo em radianos do braço
#obs[1] - Gripper
#obs[2] - X da ferramenta
#obs[3] - Y da ferramenta
#obs[4] - Z da ferramenta
#obs[5] - Angulo da ferramenta em graus
#obs[6] - Quaternion
#obs[7] - Quaternion
#obs[8] - Quaternion
###########################
#Para achar o ângulo da ferramenta relativamente ao braço, faça obs[5]-obs[0]
#Para saber se a ferramenta caiu, verificar de obs[4] > 0.07 (Não caiu)
