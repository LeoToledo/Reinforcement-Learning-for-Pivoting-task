import numpy as np
from gym import utils
from gym.envs.my_envs import mujoco_env
import matplotlib.pyplot as plt
import math


class PivotingEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    
    def __init__(self):
        self.ep = 0
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'old_pivoting.xml', 2)
        
    
    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        
        ob = self._get_obs()       
        print("AQUI: ", self.ep)
        reward = 0
        done = 0
        return ob, reward, done, {}

    def reset_model(self):
        #Recebe qpos e qvel do modelo do mujoco, com vários dados que não serão utilizados
        qpos = self.init_qpos
        qvel = self.init_qvel
        
        self.ep += 1
        
        self.set_state(qpos, qvel)
        return self._get_obs()


    def _get_obs(self):
        '''
        #Pegando o Angulo da ferramenta em radianos
        obs = self.sim.data.get_joint_qpos("tool")
        tools_angle = np.arctan2(2*(obs[3]*obs[6] + obs[4]*obs[5]), (1 - 2*(obs[5]**2 + obs[6]**2)))      
        
        #Gripper's angle
        grippers_angle = self.sim.data.get_joint_qpos("kuka_joint_6")
        #Angle between tool and gripper
        relative_angle = tools_angle - grippers_angle  
        #Gripper's Distance
        grippers_dist = obs[1]
        #Gripper's velocity
        grippers_vel = self.sim.data.get_joint_qvel("kuka_joint_6")
        #Tool's velocity relative to gripper
        tools_vel = self.sim.data.get_joint_qvel("tool")
        tools_vel = tools_vel[5] #Tool's global velocity
        tools_vel = tools_vel - grippers_vel
       
        if(obs[2] < 1.1):
            drop = 1
        else:
            drop = 0
        obs = np.concatenate([ [relative_angle], [grippers_dist], [grippers_vel], [tools_vel], [drop] ]).ravel()
        
        return obs
'''
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
    

