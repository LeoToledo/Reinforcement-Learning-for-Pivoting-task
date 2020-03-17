import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import matplotlib.pyplot as plt
import sys

class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'pivotingrika.xml', 2)
    
    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        
        ob = self._get_obs()    
     
        
        reward = 0
        done = 0
        return ob, reward, done, {}

    def reset_model(self):
        #Recebe qpos e qvel do modelo do mujoco, com vários dados que não serão utilizados
        qpos = self.init_qpos
        qvel = self.init_qvel
        
       
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        #Pega o qpos do mujoco e guarda em obs
        obs = np.concatenate([self.sim.data.qpos, [0]]).ravel()
        
        #Vamos converter o angulo da ferramenta para graus
        conv_rad = np.arctan2(2*(obs[5]*obs[8] + obs[6]*obs[7]), (1 - 2*(obs[7]**2 + obs[8]**2)))
        obs_degrees = 180*conv_rad/np.pi
        
        #Por fim, o angulo global da ferramenta convetido para graus é adicionado em obs[5]
        obs[5] = obs_degrees
        
        #Convertendo o Angulo do Gripper para Graus
        obs[0] = 180*obs[0]/np.pi

        #Aqui, calculamos o ângulo relativo em graus
        relative_angle = obs[5] - obs[0]
        
        #Aqui, vamos pegar a velocidade do gripper apenas
        vel_gripper = self.sim.data.get_joint_qvel("s_rotation")
        
        #Aqui,vamos pegar a velocidade da ferramenta em relação ao gripper
        vel_rel = self.sim.data.get_joint_qvel("fr")
        #print(vel_rel)

        #Observação para verificar se a ferramenta caiu
        #0.07 motor
        #-0.1 position
        if(obs[4] < 0.07):
            caiu = 1
        else:
            caiu = 0

        #A única observação que o ambiente deve dar é o ângulo relativo entre a ferramenta e o gripper
        obs_final = np.concatenate([ [relative_angle], [obs[1]], [vel_gripper], [vel_rel[5]-vel_gripper], [caiu] ]).ravel()
        
        return obs_final

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent + 0.7
    
   
        


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


##############################
#obs_final[0] - Angulo relativo
#obs_final[1] - Posição do gripper
#obs_final[2] - Velocidade do gripper