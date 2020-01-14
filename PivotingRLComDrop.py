import gym
from keras import models
from keras import layers
from keras.optimizers import Adam
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

########################################################################
#Para rodar com a gpu
config = tf.ConfigProto( device_count = {'GPU': 2 , 'CPU': 1} ) 
sess = tf.Session(config=config)
keras.backend.set_session(sess)
########################################################################

GAMMA = 0.99
EPSILON = 1
EPSILON_DECAY = 0.01
EPSILON_MIN = 0.01
LEARNING_RATE = 0.01

BUFFER_LEN = 200000000
NUMBER_OF_EPISODES = 10000
NUMBER_OF_ITERATIONS = 1200
PICK_FROM_BUFFER_SIZE = 48
DESIRED_ANGLE = 15

NUMBER_OF_ACTIONS = 8

class DQN_Agent:
    def __init__(self, env):
    #Definindo as variáveis
        self.env = env
        self.gamma = GAMMA
            
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN
        
        self.learning_rate = LEARNING_RATE
    
        self.replay_buffer = deque(maxlen = BUFFER_LEN)
    
        self.model_network = self.create_network()
        self.target_network = self.create_network()
        self.target_network.set_weights(self.model_network.get_weights())
            
        self.episode_num = NUMBER_OF_EPISODES
        self.iteration_num = NUMBER_OF_ITERATIONS
        self.pick_buffer_every = PICK_FROM_BUFFER_SIZE
            
        #Ações e ângulos
        self.desired_angle = DESIRED_ANGLE
            
        #Variáveis de análise
        self.total_rw_per_ep = []
        self.total_steps_per_ep = []
        
   #Modelando a rede neural
    def create_network(self):
        model = models.Sequential()
        #Pega o tamanho do espaço de observações do ambiente
        state_shape = self.env.observation_space.shape
        
        #A rede tem duas hidden layers, uma com 24 nós e outra com 48
        model.add(layers.Dense(32, activation='relu', input_shape=state_shape))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        #O tamanho da output layer é igual ao tamanho do espaço de ações
        model.add(layers.Dense(NUMBER_OF_ACTIONS, activation='linear'))
    #        print("TAMANHO DO ESPACO DE ACOES: ", len(self.env.action_space.sample()))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        
        return model
    
    #Escolhe qual ação tomar(aleatória ou não)
    def greedy_action(self, state):
        #Se atingir o epsilon min, fica nele.
        self.epsilon = max(self.epsilon_min, self.epsilon)
            
        #Escolhe um número aleatório entre 0 e 1. Se ele for menor do que epsilon, toma uma ação aleatória
        if(np.random.rand(1) < self.epsilon):
            action = np.random.randint(0, NUMBER_OF_ACTIONS)
#            print("RANDOM ACTION: ", action)
        else:
            action = np.argmax(self.model_network.predict(state)[0])
        return action
    
    def replay_memory(self):
        #Checa se o tamanho atual do buffer é menor do que o tamanho mínimo necessário
        if(len(self.replay_buffer) < self.pick_buffer_every):
            return
        
        #Pega uma amostra randomica do buffer. A amostra possui tamanho "pick_buffer_every"
        samples = random.sample(self.replay_buffer, self.pick_buffer_every)
            
        states = []
        new_states=[]
            
        #Itera em samples. Cada sample tem a forma (state, action, reward, new_state, done)
        for sample in samples:
            #Armazena a sample atual nas variáveis    
            state, action, reward, new_state, done = sample
            #Adiciona as variáveis nas listas.
            states.append(state)
            new_states.append(new_state)
            
        #Transforma a lista de estados em um array
        states_array = np.array(states) 
        #Dá um reshape, criando uma linha para cada step(para cada sample) e n colunas, uma para cada estado.
        states = states_array.reshape(self.pick_buffer_every, env.observation_space.shape[0])
        #Transforma a lista de estados em um array
        new_states_array = np.array(new_states)
        #Dá um reshape, criando uma linha para cada step(para cada sample) e n colunas, uma para cada estado.
        new_states = new_states_array.reshape(self.pick_buffer_every, env.observation_space.shape[0])
    
        #Dá um predict na model_network para pegar os Q-values atuais
        targets = self.model_network.predict(states)

        #Dá um predict na target_network para pegar os novos Q-values
        new_qs = self.target_network.predict(new_states)
    
        i = 0
        for sample in samples:
            state, action, reward, new_state, done = sample    
            #target recebe os Q-values antigos da iteração atual
            target = targets[i]
            #Caso tenha terminado, o Q-value referente à ação atual recebe "reward"
            if done and reward > 0:
                target[action] = reward
            #Caso não tenha terminado, recebe a relação do Q-Learning.
            else:
                #new_q recebe o novo Q-value da iteração atual. Note que "max" já passa o maior Q-value.
                new_q = max(new_qs[i])
                #O Target da ação da iteração atual recebe então a relação.
                target[action] = reward + new_q * self.gamma
            i+=1
            
        #Por fim, treina a model_network com os Q-values atualizados
        self.model_network.fit(states, targets, epochs=1, verbose=0)
    
#Define qual ação complexa será executada
    def check_action(self, act):
        self.check_act = []
        #Vai para a esquerda e pressiona o gripper
        if(act == 0):
            self.check_act.append(0.8)
            self.check_act.append(-0.015)
            
        #Vai para a direita e pressiona o gripper
        elif(act == 1):
            self.check_act.append(-0.8)
            self.check_act.append(-0.015)
       
        #Fica parado e pressiona o gripper
        elif(act == 2):
            self.check_act.append(0)
            self.check_act.append(-0.015)
        
        #Vai para a esquerda e não pressiona o gripper
        elif(act == 3):
            self.check_act.append(0.8)
            self.check_act.append(0)
         
        #Vai para a direita e não pressiona o gripper    
        elif(act == 4):
            self.check_act.append(-0.8)
            self.check_act.append(0)
        
        #Fica parado e não pressiona o gripper
        elif(act == 5):
            self.check_act.append(0)
            self.check_act.append(0)
            
        #Vai pouco para a esquerda e não pressiona
        elif(act == 6):
            self.check_act.append(0.4)
            self.check_act.append(0)
        
        #Vai pouco para a direita e não pressiona
        elif(act == 7):
            self.check_act.append(-0.4)
            self.check_act.append(0)
            
        return self.check_act
            
    def play(self, current_state, eps):
            reward_sum = 0
            
            #Contador que garante que a ferramenta fique parada no ângulo desejado por uma quantidade mínima de tempo
            self.desired_angle_counter = 0
            
            #Itera nos steps
            for i in range(self.iteration_num):
                
                #Escolhe o número da ação a ser tomada
                action = self.greedy_action(current_state)
                
                #Cria um vetor que irá receber o valor da ação complexa
                self.action_taken = []
                
                #Adiciona a ação complexa no vetor
                self.action_taken = self.check_action(action)
                #Agente toma a ação
                new_state, reward, done, _ = env.step(self.action_taken)
                new_state = new_state.reshape(1, env.observation_space.shape[0])
                
                 #Renderiza a cada N episódios
                if(eps%30 == 0):
                   env.render()
             
                #Definição do ângulo relativo entre o Gripper e a Ferramenta
                relative_angle = new_state[0][5] - new_state[0][0]
                #Definição dos limites aceitáveis de sucesso
                self.desired_angle_lowbound = self.desired_angle - 0.4
                self.desired_angle_highbound = self.desired_angle + 0.4
                
                #Caso a ferramenta esteja no range de ângulo desejado e não tenha caído no chão
                if(relative_angle >= self.desired_angle_lowbound and relative_angle <= self.desired_angle_highbound and reward != -2):
                    #Quanto mais tempo ele ficar na posição desejada, mais recompensa ele receberá
                    self.desired_angle_counter = self.desired_angle_counter + 1
                    reward = 1
                    
                    #Caso fique uma quantidade minima de tempo no angulo desejado, conclui o episódio
                    if(self.desired_angle_counter >= 50):
                        self.desired_angle_counter = 0
                        done = 1
                        completou = 1
                    else:
                        completou = 0
                  
                    
                #Caso a ferramenta não esteja no ângulo desejado    
                else:
#                    Caso a ferramenta não tenha caído no chão 
                    if(reward != -2):
                        reward = (-1)*np.abs(relative_angle - self.desired_angle)/180
                    completou = 0
                    
                   
                #Adiciona os dados do step no buffer
                self.replay_buffer.append([current_state, action, reward, new_state, done])
                
                #Chama o replay memory, que só é executado quando temos um buffer de tamanho aceitável.
                self.replay_memory()
                
                #Soma a reward e atualiza o estado atual
                reward_sum += reward
                current_state = new_state 
                
                #Caso tenha concluído no step atual, dá um break no loop
                if done:
                    break
            
            #Armazena a reward total e o numero total de steps gastos para, posteriormente, plotar graficos
            self.total_rw_per_ep.append(reward_sum)
            self.total_steps_per_ep.append(i)
                   
        #Checagem de sucesso ou fracasso do episodio  
            if(completou == 0):
                print("Episodio: ", eps, " - FAILED - ", i, "Steps", " || Reward: ", reward_sum, " || EPSILON: ", self.epsilon, "Angulo final: ", relative_angle)
            else:
                print("Episodio: ", eps, " - SUCESSO - ", i, "Steps", " || Reward: ", reward_sum, " || EPSILON: ", self.epsilon, "Angulo final: ", relative_angle)
                self.model_network.save('./PivotingTrainedNet', eps, 'h5')
            
            #Copia os pesos da target para a train
            self.target_network.set_weights(self.model_network.get_weights())
            
            
    def start(self):
        #Itera nos episódios
        for eps in range(self.episode_num):
            
            current_state = env.reset().reshape(1, env.observation_space.shape[0])
            self.play(current_state, eps)
            
            #Na primeiras 10 iterações, a taxa de exploração é de 100%
            if(eps <= 9 ):
                self.epsilon = 1
            else:
                #Decai o epsilon
                self.epsilon -= self.epsilon_decay

    
    def plotar_graficos(self):
    #Plotando a reward
        plt.plot(self.total_rw_per_ep, color='g')
        plt.ylabel("Reward total")
        plt.xlabel("Episodio")
        plt.savefig("Rewards Plot")
        plt.close()
        
    #Plotando o número de steps gasto
        plt.plot(self.total_steps_per_ep, color='b')
        plt.ylabel("Total de Steps")
        plt.xlabel("Episodio")
        plt.savefig("Steps Plot")
        plt.close()
            
                    
env = gym.make("Pivoting-v0")
env._max_episode_steps = 1200
dqn = DQN_Agent(env)
dqn.start()
    
    
dqn.plotar_graficos()
                    
                    
                
           
        
        
        
        
        
        
        
        
    
        
        
        
        
        
        
        
        
            
            
            
