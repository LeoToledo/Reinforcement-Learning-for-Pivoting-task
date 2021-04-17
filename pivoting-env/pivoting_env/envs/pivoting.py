import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import matplotlib.pyplot as plt
import math
from controllers_utils import CtrlUtils

MAX_EP_LEN = 1000
N_JOINTS = 7
JOINT_6 = 5
JOINT_3 = 3

class PivotingEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        self.ep = 0
        self.current_ep = 0
        self.counter = 0
        self.desired_angle = 0
        self.erro = 0
        self.current_step = 0
        
        self.ep_ret = 0
        self.ep_len = 0
        self.ctrl = None
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'pivoting_kuka.xml', 8)


    def ctrl_action_torque(self, a, qposd_robot):
        '''Métodos de torque'''

        # método 1: nessa config o PPO vai atuar JUNTO com meu controlador em torque EM TODAS AS JUNTAS
        #u = self.ctrl.ctrl_action(self.sim) + a[:N_JOINTS]

        #método 2: nessa config o PPO vai controlar tudo em torque sem ajuda do meu controlador
        #u = a[:N_JOINTS]

        # método 3: aqui o PPO vai atuar só nas juntas "planares" em torque
        u = self.ctrl.ctrl_action(self.sim)
        u[JOINT_6] += a[JOINT_6]
        u[JOINT_3] += a[JOINT_3]

        return u

    # def ctrl_action_position(self, a):
    #     '''Métodos de posição'''
    #     # método 1: nessa config o PPO vai atuar JUNTO com meu controlador em posição em TODAS AS JUNTAS
    #     # qposd_ppo = np.array([0, 0, 0, -np.pi / 2, -np.pi / 2, 0, 0]) + a[:N_JOINTS]
    #
    #     # método 2: nessa config o PPO vai controlar tudo em posição. Aprendizado mais lento!
    #     # qposd_ppo = a[:N_JOINTS]
    #
    #     # método 3: aqui o PPO vai atuar só nas juntas "planares" em posição
    #     # qposd_ppo = np.array([0, 0, 0, -np.pi / 2, -np.pi / 2, 0, 0])
    #     # qposd_ppo[JOINT_6] += a[JOINT_6]
    #     # qposd_ppo[JOINT_3] += a[JOINT_3]
    #
    #     return qposd_ppo


    def step(self, a):
        if self.ctrl is None:
            self.ctrl = CtrlUtils(self.sim)
        # self.do_simulation(a, self.frame_skip)
        qposd_robot = np.array([0, 0, 0, -np.pi / 2, -np.pi / 2, 0, 0])

        # TODO: NAO USAR ESSES AQUI POR ENQUANTO, TEMOS QUE ALTERAR O XML
        # PPO ATUANDO NA POSICAO (comentar aqui se for usar torque)
        # qpos_ppo = self.ctrl_action_position(a)
        # self.ctrl.calculate_errors(self.sim, qpos_ref=qpos_ppo)
        # u = self.ctrl.ctrl_action(self.sim)

        # PPO ATUANDO NO TORQUE
        self.ctrl.calculate_errors(self.sim, qpos_ref=qposd_robot)
        u = self.ctrl_action_torque(a, qposd_robot=qposd_robot)

        # NAO EDITAR DAQUI PRA BAIXO
        self.sim.data.ctrl[:self.ctrl.nv] = u  # jogando ação de controle nas juntas
        self.sim.data.ctrl[-1] = a[-1] # ação de controle na garra. Esse aqui é tudo por conta do PPO
        self.sim.step()	
        #if(self.ep % 20 and self.current_step%2 == 0):
            #self.sim.render(mode="window")
        ob = self._get_obs()
        
        #Modelando a reward
        #Primeiramente, pegamos o target e o erro
        desired_angle = self.desired_angle
        
        erro = self.erro
        self.current_step += 1    
        #Angulo relativo entre a ferramenta e o gripper       
        angle_error = ob[0]
        current_angle = angle_error + desired_angle
        # print("ANGULO ATUAL: ", current_angle)
        #Checa se está na região de sucesso
        if(current_angle >= (desired_angle - erro) and current_angle <= (desired_angle + erro) ): 
            reward = (-1)*np.abs(current_angle - desired_angle)/200
            self.counter = self.counter + 1
            done = 0

            #Caso fique uma quantidade de tempo na região de sucesso
            if(self.counter > 120):
                print("*********************Completou***********************")
                print(" || Episodio: ", self.ep," || Reward: ", np.round(self.ep_ret, 0), " || Steps: ", self.ep_len, 
                    " || Angulo Final: ", np.round(angle_error + desired_angle, 3), " || Target: ", desired_angle)
                reward = 10
                
                #Zerando o contador de tempo na zona de sucesso
                self.counter = 0
                #Zerando o tempo do episódio
                self.ep_len = 0
                #Zerando a recompensa acumulada
                self.ep_ret = 0
                
                done = 1
                return ob, reward, done, {}
            
        #Se não completar
        else:
            self.counter = 0
            reward = (-1)*np.abs(current_angle - desired_angle)/100
            done = 0
        
        #Reward total e duração do episodio
        self.ep_ret += reward
        self.ep_len += 1
        
        if(self.current_step == MAX_EP_LEN):
            #Printando resultados
            print(" || Episodio: ", self.ep," || Reward: ", np.round(self.ep_ret, 0), " || Steps: ", self.ep_len, 
                    " || Angulo Final: ", np.round(angle_error + desired_angle, 3), " || Target: ", desired_angle)
            #Zerando a duração do episodio
            self.ep_len = 0
            #Atualizando o episodio atual
            self.current_ep += 1
            #Zerando a recompensa acumulada
            self.ep_ret = 0
        
        return ob, reward, done, {}

    def reset_model(self):
        # print("epoch =", self.ep)
        #Recebe qpos e qvel do modelo do mujoco, com vários dados que não serão utilizados
        qpos_init_robot = [0, 0, 0, -np.pi / 2, -np.pi / 2, 0, 0]
        qpos_init_gripper = [0]
        qpos_init_tool = [0.04, 0, 1.785, 1, 0, 0 ,0]
        qpos = np.concatenate((qpos_init_robot, qpos_init_gripper, qpos_init_tool))
        qvel = self.init_qvel

	#Atualizando o numero do episódio e steps atuais. Está com gambiarra
        self.ep += 1       
        self.counter = 0
        self.current_step = 0

	#Definindo o angulo desejado(target)
        self.desired_angle = np.random.randint(-30, 30)
        while(self.desired_angle == 0):
            self.desired_angle = np.random.randint(-30, 30)
        
        #Definindo o erro aceitável para a conclusão do objetivo
        self.erro = min(np.abs(self.desired_angle/7), 3)
        if( (self.desired_angle) <= 6 or (self.desired_angle) >= -6):
            self.erro = self.erro + 0.3

        self.set_state(qpos, qvel)
        return self._get_obs()


    def _get_obs(self):

        #Pegando o Angulo da ferramenta em radianos
        obs = self.sim.data.get_joint_qpos("tool")
        tools_angle = np.arctan2(2*(obs[3]*obs[6] + obs[4]*obs[5]), (1 - 2*(obs[5]**2 + obs[6]**2)))
        tools_angle = 180*tools_angle/np.pi
        
        #Gripper's angle
        grippers_angle = (-1)*(self.sim.data.get_joint_qpos("kuka_joint_6"))
        grippers_angle = 180*grippers_angle/np.pi
        
        #Angle error between tool and gripper with relation to the desired angle
        relative_angle = tools_angle - grippers_angle - self.desired_angle
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


    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent + 0.7


