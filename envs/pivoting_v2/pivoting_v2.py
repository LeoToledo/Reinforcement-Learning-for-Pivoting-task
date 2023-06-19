import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scripts.mujoco_utils import MujocoUtilsCartesian
from scripts.kuka_impedance import KukaImpedance

FRANKA_WORKSPACE = 0.855
MAX_SIM_TIME = 2.0

class PivotingEnv(gym.Env):
    def __init__(self, states=['fz'], actions=['z'], path_to_model='', show_sim=True, show_frames=True):
        super(PivotingEnv, self).__init__()
        
        # Define action and observation space
        # They must be gym.spaces objects
        n_actions = len(actions)
        n_states = len(states)
        self.action_space = spaces.Box(low=np.array([-1]*n_actions, dtype=np.float32), high=np.array([1]*n_actions, dtype=np.float32), 
                                        shape=(n_actions,), dtype=np.float32)

        self.observation_space = spaces.Box(low=np.array([-100]*n_states, dtype=np.float64), high=np.array([100]*n_states, dtype=np.float64),
                                            shape=(n_states,), dtype=np.float64)
        
        self.states_keys = states
        self.actions_keys = actions
        
        self.dt_rl = 50/1000  # ms
        self._nu = 7

        # Mujoco defs
        self.mj = MujocoUtilsCartesian(ee_site_name='ee_site', tool_name='flat_tool', render=show_sim, render_cartesian_frames=show_frames)
        self.mj.load_model_mujoco(path_to_model=path_to_model)
        self.dt = self.mj.model.opt.timestep
        
        
        # self.xd = np.zeros(3)
        # self.xdmat = np.zeros((3,3))
        self.x = np.zeros(3)
        self.xmat = np.zeros((3,3))
        self.x0 = np.zeros(3)
        self.x0mat = np.zeros((3,3))

        self.xtgt = np.zeros(3)
        self.xmattgt = np.zeros((3,3))
        self.x0tgt = np.zeros(3)
        self.x0mattgt = np.zeros((3,3))

        self.xobj = np.zeros(3)
        self.xmatobj = np.zeros((3,3))

        self.quat0 = np.zeros(4)
        # self.qpos0 = np.array([0, 0.25, 0, -1.4, 0, 1.6, 0.785])
        self.qpos0 = np.array([0, -0.25, 0, -1.8, 0, 1.6, 0.785])

        self.ctrl = KukaImpedance()
        self.ctrl.set_bk()  # change this to init controller

    def step(self, action):
        # we can have actions in differente domains: forces, positions...
        # so I decided to keep the network outputting [-1,1], but them we manually remap each action
        action = self.remap_actions(action)

        self.x, self.xmat       = self.mj.get_robot_xpose()
        self.xobj, self.xmatobj = self.mj.get_object_xpose()
        self.xtgt, self.xmattgt = self.mj.get_target_xpose()

        # example action: this one I want to work in Cartesian position
        if self.xobj[2] < 0.2:
            flag_action = 0
        else:
            flag_action = 1
        xd = self.x+action[:3]*flag_action # TODO: add filter
        # xdmat = self.xmat+R.from_euler('zyx',np.array(action[3:]), degrees=True).as_matrix()        

        # get the joint torques and apply them to the robot
        self.ctrl.calculate_errors(self.mj, xd=xd, xdmat=self.x0mat)
        tau = self.ctrl.tau_impedance(self.mj)
        self.mj.set_joint_torques(tau)

        # get the observation from the previous step
        obs = self.get_obs()

        # TODO: reward engineering
        reward = self.get_reward()

        # done
        done = False

        # if self.mj.sim.data.time >= MAX_SIM_TIME:
        #     done = True
        #     reward -= 500

        if self.get_distance_tgt_obj_normalized() < 0.1:# and np.linalg.norm(self.mj.get_object_xvel()) < 0.005:
            done = True
            reward += 500
        
        if np.linalg.norm(self.mj.get_object_xvel()) < 0.001 and self.mj.get_object_xpose()[0][2] < 0.2:
            done = True
            # reward -= (MAX_SIM_TIME - self.mj.sim.data.time)/0.001
            reward += 1/(self.get_distance_tgt_obj_normalized())
        
        # example done 2: reached other objective
        # if self.mj.
        
        # TODO: I don't know what this is may be used for
        info = {}
        truncated = False  # TODO: what is truncated?

        # advance mj sim a whole step
        self.mj.step_sim()

        return obs, reward, done, truncated, info

    def reset(self):
        # full reset of mj model forcefully initial joint pos as self.qpos_init
        self.mj.reset_sim()
        self.mj.set_robot_qpos(self.qpos0)
        self.mj.forward()
        self.mj.set_object_xpos(random_pose=False)
        self.mj.set_target_xpos(random_pose=False)

        self.x0, self.x0mat = self.mj.get_robot_xpose()
        self.xtgt, self.xmattgt = self.mj.get_target_xpose()
        self.x0obj, self.x0matobj = self.mj.get_object_xpose()

        # this is demanded by Gym
        observation = self.get_obs()

        # possible useful link: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#using-callback-monitoring-training

        return observation, {}  # reward, done, info can't be included

    # TODO: update obs space
    def get_obs(self):
        # states = ['fx', 'fy', 'fz', 'tx', 'ty', 'tz', 
        #           'd_ee_obj_x', 'd_ee_obj_y', 'd_ee_obj_z',
        #           'd_ee_tgt_x', 'd_ee_tgt_y', 'd_ee_tgt_z']
        # ft = self.mj.get_ft()
        d_ee_obj = self.x - self.xobj
        d_ee_tgt = self.x - self.xtgt
        # obs = np.concatenate([ft, d_ee_obj, d_ee_tgt])
        obs = np.concatenate([d_ee_obj, d_ee_tgt])
        return np.array(obs, dtype=np.float64)
    
    def get_reward(self):
        # ft = -np.linalg.norm(self.mj.get_ft())/(10)*0
        # d_tgt_obj = -np.linalg.norm(self.xtgt - self.xobj)/np.linalg.norm(self.x0tgt - self.x0obj)
        # d_tgt_obj = -np.linalg.norm(self.xtgt[:2] - self.xobj[:2])/np.linalg.norm(self.x0tgt[:2] - self.x0obj[:2])
        d_tgt_obj = self.get_distance_tgt_obj_normalized()
        d_x_x0 = np.linalg.norm(self.x - self.x0)/(2*FRANKA_WORKSPACE)
        reward = - d_tgt_obj - d_x_x0
        return reward

    # TODO: update remapping. I always prefer to have a clipped [-1,1] for the network, then we do the remap ourselves.
    def remap_actions(self, action):
        # TODO: everything here needs to be scaled
        action[:3] = action[:3]*FRANKA_WORKSPACE
        # action[3:] = action[3:]*30*np.pi/180
        return action
    
    def get_distance_tgt_obj_normalized(self):
        return np.linalg.norm(self.xtgt[:2] - self.xobj[:2])/np.linalg.norm(self.x0tgt[:2] - self.x0obj[:2])