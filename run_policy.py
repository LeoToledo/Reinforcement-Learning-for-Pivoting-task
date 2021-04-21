from spinup.utils.test_policy import load_policy_and_env, run_policy
import pivoting_env
import gym
from datetime import datetime

# Save path
SAVE_FILE_NAME = 'teste1'
today = datetime.today().strftime('%Y-%m-%d')
PATH = 'data/' + today + '_' + SAVE_FILE_NAME

_, get_action = load_policy_and_env(PATH)
env = gym.make('pivoting-v0')
run_policy(env, get_action)