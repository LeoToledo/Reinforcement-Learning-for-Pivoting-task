import gym
import tensorflow as tf
from spinup import ppo_tf1 as ppo
import pivoting_env

env_fn = lambda: gym.make('pivoting_env:pivoting-v0')

logger_kwargs = dict(output_dir='~/First-Pivoting-Model-2019/.', exp_name='teste1')

ac_kwargs = dict(hidden_sizes=[64, 64], activation=tf.nn.relu)

ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=250, logger_kwargs=logger_kwargs)


# from spinup.utils.test_policy import load_policy_and_env, run_policy
# import your_env
# _, get_action = load_policy_and_env('/path/to/output_directory')
# env = your_env.make()
# run_policy(env, get_action)