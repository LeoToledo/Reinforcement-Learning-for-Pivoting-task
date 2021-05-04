from spinup import ppo_tf1 as ppo
import tensorflow as tf
import gym
import pivoting_env
from datetime import datetime

# Simulation parameters
STEPS_PER_EPOCH = 4000
EPOCHS = 4000

# Save path
SAVE_FILE_NAME = '3000epochs-full-action-space-2'
today = datetime.today().strftime('%Y-%m-%d')
PATH = 'data/' + today + '_' + SAVE_FILE_NAME 

# Environment name
env_fn = lambda : gym.make('pivoting-v0')

# MLP parameters
ac_kwargs = dict(hidden_sizes=[128,64,24,6], activation=tf.nn.relu)

logger_kwargs = dict(output_dir=PATH, exp_name='3000epochs-full-action-space-2')

# PPO RUN 
ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch= STEPS_PER_EPOCH, epochs= EPOCHS, logger_kwargs=logger_kwargs)
