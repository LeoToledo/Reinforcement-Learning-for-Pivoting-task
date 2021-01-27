import gym
import tensorflow as tf
from spinup import ppo_tf1 as ppo

env_fn = lambda: gym.make('pivoting_env:pivoting-v0')

logger_kwargs = dict(output_dir='.', exp_name='teste1')

ac_kwargs = dict(hidden_sizes=[64, 64], activation=tf.nn.relu)

ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=250, logger_kwargs=logger_kwargs)