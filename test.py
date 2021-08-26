import time
import gym
from PPO.PPO import PPO
import numpy as np
import pivoting_env
import yaml

# Read YAML file
with open('./parameters.yaml', 'r') as file_descriptor:
    parameters = yaml.load(file_descriptor)


def test():
    global done
    print("============================================================================================")

    ################## hyperparameters ##################
    env_name = parameters['model']['env_name']
    has_continuous_action_space = True
    max_ep_len = parameters['model']['max_ep_len']  # max timesteps in one episode
    action_std = parameters['ppo']['action_parameters'][
        'min_action_std']  # set same std for action distribution which was used while saving

    render = parameters['test']['render']  # render environment on screen
    frame_delay = 0  # if required; add delay b/w frames

    total_test_episodes = parameters['test']['number_of_test_episodes']  # total num of testing episodes

    K_epochs = parameters['ppo']['hyperparameters']['k_epochs']  # update policy for K epochs
    eps_clip = parameters['ppo']['hyperparameters']['eps_clip']  # clip parameter for PPO
    gamma = parameters['ppo']['hyperparameters']['gamma']  # discount factor

    lr_actor = parameters['agent']['mlp']['lr_actor']  # learning rate for actor
    lr_critic = parameters['agent']['mlp']['lr_critic']  # learning rate for critic
    #####################################################

    env = gym.make(env_name)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = len(parameters['model']['ppo_acting_joints'])
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)

    # preTrained weights directory

    random_seed = 42  #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0  #### set this to load a particular checkpoint num

    directory = "PPO/PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + parameters['test']['model_name']
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes + 1):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len + 1):

            #####################
            state[[3]] = state[[3]] * 100
            #####################

            action = ppo_agent.select_action(state)

            ####!!!!!@#####
            action = rescale_action_space(scale_factor=15, action=action)
            #######

            state, reward, done, _ = env.step(action)
            ep_reward += reward

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        print(
            f"Episode : {ep} \t Average Reward : {int(ep_reward)} \t Real : {int(env.get_current_angle())} "
            f"\t Target : {env.get_desired_angle()} \t Sucess : {done - env.get_drop_bool()}")

    env.close()


def rescale_action_space(scale_factor, action):
    action_temp = np.zeros(8)
    action_temp[parameters['model']['ppo_acting_joints']] = action * scale_factor

    if action_temp[7] > 0:
        action_temp[7] = 25
    else:
        action_temp[7] = -25
    return action_temp


if __name__ == '__main__':
    test()
