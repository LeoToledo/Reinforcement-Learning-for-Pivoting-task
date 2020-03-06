import time
import joblib
import os
import os.path as osp
import tensorflow as tf
import torch
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph
import numpy as np
import matplotlib.pyplot as plt

def load_policy_and_env(fpath, itr='last', deterministic=False):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the 
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a 
    PyTorch save.
    """

    # determine if tf save or pytorch save
    if any(['tf1_save' in x for x in os.listdir(fpath)]):
        backend = 'tf1'
    else:
        backend = 'pytorch'

    # handle which epoch to load from
    if itr=='last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        if backend == 'tf1':
            saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x)>8]

        elif backend == 'pytorch':
            pytsave_path = osp.join(fpath, 'pyt_save')
            # Each file in this folder has naming convention 'modelXX.pt', where
            # 'XX' is either an integer or empty string. Empty string case
            # corresponds to len(x)==8, hence that case is excluded.
            saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x)>8 and 'model' in x]

        itr = '%d'%max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d'%itr

    # load the get_action function
    if backend == 'tf1':
        get_action = load_tf_policy(fpath, itr, deterministic)
    else:
        get_action = load_pytorch_policy(fpath, itr, deterministic)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


def load_tf_policy(fpath, itr, deterministic=False):
    """ Load a tensorflow policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'tf1_save'+itr)
    print('\n\nLoading from %s.\n\n'%fname)

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, fname)

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x : sess.run(action_op, feed_dict={model['x']: x[None,:]})[0]

    return get_action


def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""
    
    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0

    #############################
    desired_angle = 20
    counter = 0
    angulo_store = []
    master_angle_store = []
    #############################

    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-10)

        a = get_action(o)
        o, r, d, _ = env.step(a)
        
#######################################################################################################################3
        #Atualizando a reward
        #Caso complete o objetivo
        if(o[0] >= (desired_angle - np.abs(desired_angle/8 + 0.2)) and o[0] <= (desired_angle + np.abs(desired_angle/8 + 0.2))): 
            r = (-1)*np.abs(o[0] - desired_angle)/100
            counter = counter + 1
            d = 0

            #Caso fique uma quantidade de tempo na região de sucesso
            if(counter > 100):
                print("*********************Completou***********************")
                r = 1
                counter = 0
                d = 1
        #Se não completar
        else:
            counter = 0
            r = (-1)*np.abs(o[0] - desired_angle)/100
            d = 0

        #Atualizando o ângulo relativo para um erro entre ele e o desejado
        o[0] = o[0] - desired_angle
#######################################################################################################################3
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            #Guardando o ângulo
            angulo_store.append(o[0]+desired_angle)

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episodio %d \t Reward %.3f \t EpLen %d \t Angulo %.1f \t Target %.1f'%(n, ep_ret, ep_len, (o[0] + desired_angle), desired_angle))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1
            if(n == 100):
                angulo0_store = angulo_store
                angulo_store = []
                desired_angle = 10
            elif(n == 200):
                angulo1_store = angulo_store
                angulo_store = []
                desired_angle = 15
            elif(n == 300):
                angulo2_store = angulo_store
                angulo_store = []
                desired_angle = 20
            elif(n == 400):
                angulo3_store = angulo_store
                angulo_store = []
                
     
    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()
########################################################
    #Plotando steps antes do sucesso
    #Ângulos
    plt.figure(num=None, figsize=(20, 12), dpi=120, facecolor='w', edgecolor='k')
    angulo_store = [
        angulo0_store, 
        angulo1_store,
        angulo2_store,
        angulo3_store,
    ]
    
    plt.title('BoxPlot of Reached Angles With 100 Steps')
    plt.yticks(np.arange(4, 21, step=1))
    plt.ylim(4, 21)
    #labels = ['-20', '-15', '-10', '-5']
    labels = ['5','10','15','20']

    plt.boxplot(angulo_store, labels=labels, showfliers=False)
    plt.grid(True, linestyle='-', which='major', color='lightgrey',alpha=0.5)
    plt.ylabel("Reached Angle")
    plt.xlabel("Desired Angle")
    plt.savefig("/home/kodex/rl/spinningup/data/AngulosBox")
    plt.close()
########################################################

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=400)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action = load_policy_and_env(args.fpath, 
                                          args.itr if args.itr >=0 else 'last',
                                          args.deterministic)
    run_policy(env, get_action, 2000, args.episodes, not(args.norender))
