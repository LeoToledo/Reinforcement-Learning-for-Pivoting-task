import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        ##################################################################
        self.desired_angle = 30
        self.counter = 0
        self.new_angle_counter = 0
        ##################################################################
        
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'pivoting.xml', 2)
    
    def step(self, a):
        self.new_angle_counter += 1
        if(self.new_angle_counter > 4000000):
            self.desired_angle = 45
        elif(self.new_angle_counter > 8000000):
            self.desired_angle = 15
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()    
        
    #Modelando a reward
        #Caso complete o objetivo
        if(ob[0] >= self.desired_angle - 0.5 and ob[0] <= self.desired_angle + 0.5): 
            #O denominador da reward busca aumentar a precisão, enquanto o numerador aliado aos ifs 
            reward = (-1)*np.abs(ob[0] - self.desired_angle)/180
            '''
            #busca aumentar o tempo de permanência no ângulo desejado
            if(self.counter >= 0 and self.counter <= 20):
                reward = (-1)*np.abs(ob[0] - self.desired_angle)/10
            if(self.counter > 20 and self.counter <=60):
                reward = (-1)*np.abs(ob[0] - self.desired_angle)/20
            '''
            
            self.counter = self.counter + 1
            done = 0
            if(self.counter > 60):
                reward = 100
                print("******************Completou********************")
                done = 1
                self.counter = 0
                return ob, reward, done, {}
        #Se não completar
        else:
            self.counter = 0
            reward = (-1)*np.abs(ob[0] - self.desired_angle)/180
            done = 0
        #print("REWARD: ", reward)
        ####
        return ob, reward, done, {}

    def reset_model(self):
        #Recebe qpos e qvel do modelo do mujoco, com vários dados que não serão utilizados
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
        #print("ANGULO DA FERRAMENTA: ", obs[5])
        
        #Convertendo o Angulo do Gripper para Graus
        obs[0] = 180*obs[0]/np.pi
        #print("ANGULO DO GRIPPER: ", obs[0])
        
        #obs[0] será o ângulo relativo em graus
        obs[0] = obs[5] - obs[0]
        #print("ANGULO RELATIVO: ", obs[0])
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

