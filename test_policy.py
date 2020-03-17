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
    desired_angle = -30
    counter = 0
    angulo_store = []
    master_angle_store_mean = []
    master_angle_store_max = []
    master_angle_store_min = []
    end_of_batch = 0
    #############################

    while n < num_episodes:
        #if render:
            #env.render()
            #time.sleep(1e-10)

        a = get_action(o)
        o, r, d, _ = env.step(a)
#######################################################################################################################3
        #Atualizando a reward
        #Caso complete o objetivo
        erro = min(np.abs(desired_angle/7), 3)
        if( (desired_angle) <= 6 or (desired_angle) >= -6):
            erro = erro + 0.3

        if(o[0] >= (desired_angle - erro) and o[0] <= (desired_angle + erro)): 
            r = (-1)*np.abs(o[0] - desired_angle)/200
            counter = counter + 1
            d = 0

            #Caso fique uma quantidade de tempo na região de sucesso
            if(counter > 120):
                print("*********************Completou***********************")
                r = 10
                counter = 0
                d = 1
        #Se não completar
        else:
            counter = 0
            r = (-1)*np.abs(o[0] - desired_angle)/100
            d = 0
        
        #Atualizando o ângulo relativo para um erro entre ele e o desejado
        ang_atual = o[0]
        o[0] = o[0] - desired_angle
#######################################################################################################################3
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            #Guardando o ângulo
            angulo_store.append(np.abs(desired_angle - ang_atual))

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episodio %d \t Reward %.3f \t EpLen %d \t Angulo %.1f \t Target %.1f'%(n, ep_ret, ep_len, (ang_atual), desired_angle))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1
            
            #Atualizando o ângulo desejado
            end_of_batch = end_of_batch + 1
            #Após ficar 30 vezes no mesmo ângulo, troca de angulo e salva a media do atual para plota-lo
            if(end_of_batch == 30):
                end_of_batch = 0
                #Fazendo a média dos 30 angulos utilizados
                master_angle_store_mean.append(np.mean(angulo_store))
                master_angle_store_max.append(np.max(angulo_store))
                master_angle_store_min.append(np.min(angulo_store))
                angulo_store = []

                if(desired_angle <= 30):
                    desired_angle = desired_angle + 1
                    if(desired_angle == 0):
                        desired_angle = desired_angle + 1
                else:
                    break

    #np.savetxt("master_angle_store.txt", master_angle_store, fmt="%s")          
    print("Mean: ", master_angle_store_mean)
    #print("Min: ", master_angle_store_min)
    #print("Max: ", master_angle_store_max)
    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=1800)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action = load_policy_and_env(args.fpath, 
                                          args.itr if args.itr >=0 else 'last',
                                          args.deterministic)
    run_policy(env, get_action, 4000, args.episodes, not(args.norender))
