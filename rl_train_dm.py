#!/bin/sh

import os
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from stable_baselines3 import PPO as drl_algo
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from envs.pivoting_v2.pivoting_v2 import PivotingEnv
from argparse import ArgumentParser
import re

if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')


def model_train_from_scratch(name_model, log_folder, max_timesteps, policy, env):
    drl = drl_algo(policy, env, verbose=1, tensorboard_log=log_folder)

    print('Model loaded:', name_model)
    print('Max timesteps:', max_timesteps)
    print('Chosen policy:', policy)
    return drl

def learner(name_model, log_folder, transfer_learning_folder, max_timesteps, policy, overwrite_model, render, env):
    

    user = os.environ['USER']
    if 'PBS_O_WORKDIR' in os.environ or 'WORKDIR' in os.environ:
        workdir = '/work/' + user
    else:
        workdir = '/home/' + user

    if log_folder is None:
        log_folder = Path(workdir+'/log/rl_learner_' + datetime.now().strftime("%Y%m%d-%H%M%S") +'/')
        print('Folder not defined. Creating a new logging folder')
    else:
        log_folder = Path(workdir + '/log/rl_' + log_folder)

    
    ###
    # states = ['fx', 'fy', 'fz', 'tx', 'ty', 'tz', 'd_ee_obj_x', 'd_ee_obj_y', 'd_ee_obj_z', 'd_ee_tgt_x', 'd_ee_tgt_y', 'd_ee_tgt_z']
    states = ['d_ee_obj_x', 'd_ee_obj_y', 'd_ee_obj_z', 'd_ee_tgt_x', 'd_ee_tgt_y', 'd_ee_tgt_z']
    actions = ['x', 'y', 'z']#, 'rx', 'ry', 'rz']
    path_model = name_model + '.xml'
    # if env == 'repositioning':
    env = PivotingEnv(actions=actions, states=states, path_to_model=path_model, show_sim=render)    
    # else:
    #     raise Exception('Please choose your environment correctly.')
    
    # TODO: this folder check is really bad. Update this later
    if log_folder.is_dir() and not overwrite_model:  # if folder exists, we load the model recorded there
        print('Model folder exists! Checking if there are zip files.')

        # getting all zip files
        zips = [f for f in log_folder.glob('*') if f.is_file()]

        if len(zips) > 0:
            print('Model exists! Loading last checkpoint and training for more iterations.')
            name_zip = None
            max_epochs = 0
            for i, zip in enumerate(zips):
                current = int(re.findall(r'\d+', str(zip))[-1])
                if max_epochs < current:
                    max_epochs = current
                    name_zip = zips[i]

            drl = drl_algo.load(log_folder/name_zip, tensorboard_log=log_folder)
            drl.set_env(env=env)
        else:
            drl = model_train_from_scratch(name_model, log_folder, max_timesteps, policy, env)
    else:  # if doesn't exist, we create it
        log_folder.mkdir(parents=True, exist_ok=True)
        drl = model_train_from_scratch(name_model, log_folder, max_timesteps, policy, env)
        
    checkpoint_cb = CheckpointCallback(save_freq=1e5, save_path=log_folder)

    # while True:
    drl.learn(total_timesteps=max_timesteps, reset_num_timesteps=False, callback=checkpoint_cb)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', '-mn', type=str, default='envs/pivoting_v2/assets/pivoting_kuka')
    parser.add_argument('--log_folder', '-lf', type=str)
    parser.add_argument('--transfer_learning_folder', '-tlf', type=str, default='')
    parser.add_argument('--environment', '-env', type=str, default="repositioning")
    parser.add_argument('--max_timesteps', '-mts', type=int, default=1e6)
    parser.add_argument('--policy', type=str, default='MlpPolicy')
    parser.add_argument('--overwrite_model', '-om', type=bool, default=False)
    parser.add_argument('--render', '-r', type=bool, default=True)
    args = parser.parse_args()
    learner(name_model=args.model_name,
            log_folder=args.log_folder,
            transfer_learning_folder=args.transfer_learning_folder,
            max_timesteps=args.max_timesteps,
            policy=args.policy,
            overwrite_model=args.overwrite_model,
            render=args.render,
            env=args.environment,
            )
