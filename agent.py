from collections import deque
import copy

import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend as K
from tensorflow import math as T

class Agent:
    def __init__(self):
        self.minibatch_size = 64
        self.replay_memory_size = 5000
        self.learning_rate = 0.001
        self.discount_factor = 0.75
        self.exploration = 0.1

        # replay memoly
        self.D = deque(maxlen=self.replay_memory_size)
        self.good_action_experience = deque(maxlen=self.replay_memory_size)

        # action_model
        self.init_action_model()

        # reward_model
        self.init_reward_model()

        # variables
        self.current_loss = 0.0

    def init_action_model(self):
        # activation = keras.layers.LeakyReLU()
        self.action_model = Sequential()
        self.action_model.add(Dense(100, input_shape=(4,), activation="relu"))
        self.action_model.add(Dropout(0.3))
        self.action_model.add(Dense(200, activation="relu"))
        self.action_model.add(Dropout(0.3))
        self.action_model.add(Dense(100, activation="relu"))
        self.action_model.add(Dropout(0.3))
        self.action_model.add(Dense(2))
        self.action_model.compile(
            loss="cosine_proximity",
            optimizer="adam",
        )
    
    def init_reward_model(self):
        # activation = keras.layers.LeakyReLU()
        self.reward_model = Sequential()
        self.reward_model.add(Dense(100,input_shape=(6,), kernel_initializer=keras.initializers.Zeros(), activation="relu"))
        self.action_model.add(Dropout(0.3))
        self.reward_model.add(Dense(200, kernel_initializer=keras.initializers.Zeros(), activation="relu"))
        self.action_model.add(Dropout(0.3))
        self.reward_model.add(Dense(100, kernel_initializer=keras.initializers.Zeros(), activation="relu"))
        self.action_model.add(Dropout(0.3))
        self.reward_model.add(Dense(1))
        self.reward_model.compile(
            loss="mean_squared_error",
            optimizer="adam",
        )
    
    def reward(self, state):
        return self.reward_model.predict(state)
    
    def action(self, state):
        a = self.action_model.predict(state)
        return a

    def store_experience(self, state, action, reward, state_1, terminal):
        self.D.append((state, action, reward, state_1, terminal))

    
    def _cos_sim(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    def experience_replay(self):
        reward_state_minibatch = []
        reward_y_minibatch = []
        minibatch_size = min(len(self.D), self.minibatch_size)
        minibatch_indexes = np.random.randint(0, len(self.D), minibatch_size)
        
        for j in minibatch_indexes:
            state_j, action_j, reward_j, state_j_1 ,terminal = self.D[j]

            # make input
            reward_state_j = np.hstack((state_j, action_j))
            action_j_1 = self.action(state_j_1)
            reward_state_j_1 = np.hstack((state_j_1, action_j_1))

            y_j_now = self.reward_model.predict(reward_state_j_1)
            
            # predict reward
            if terminal:
                y_j = reward_j
                if y_j > 0:
                    self.good_action_experience.append(copy.deepcopy(self.D[j]))
            else:
                # reward_j + gamma * max_action' Q(state', action')
                y_j = reward_j + self.discount_factor * y_j_now  # NOQA
                y_j = np.clip(y_j, -1.0, 1.0)
                y_j_now = np.clip(y_j_now, -1.0, 1.0)

                # check good action
                if y_j > 0 and y_j_now > y_j:
                    self.good_action_experience.append(copy.deepcopy(self.D[j]))
            #print("now state reward:{}".format(y_j_now))
            #print("update reward:{}".format(y_j))

            # make memory
            # reward memory
            if reward_state_minibatch == []:
                reward_state_minibatch = reward_state_j
            else:
                reward_state_minibatch = np.vstack((reward_state_minibatch, reward_state_j))
            if reward_y_minibatch == []:
                reward_y_minibatch = y_j
            else:
                reward_y_minibatch = np.vstack((reward_y_minibatch, y_j))
        #_reward_state_minibatch = self.normalize(reward_state_minibatch)
        self.reward_model.fit(reward_state_minibatch, reward_y_minibatch, epochs=200, verbose=0)


    def good_action_replay(self):
        action_state_minibatch = []
        action_y_minibatch = []
        minibatch_size = min(len(self.D), self.minibatch_size)
        minibatch_indexes = np.random.randint(0, len(self.good_action_experience), minibatch_size)
        
        for j in minibatch_indexes:
            state_j, action_j, _, _, _ = self.good_action_experience[j]
            if action_state_minibatch == []:
                #state_j = keras.utils.normalize(state_j)
                action_state_minibatch = state_j
            else:
                action_state_minibatch = np.vstack((action_state_minibatch, state_j))
            if action_y_minibatch == []:
                action_y_minibatch = action_j
            else:
                action_y_minibatch = np.vstack((action_y_minibatch, action_j))
        
        self.action_model.fit(action_state_minibatch,action_y_minibatch, epochs=100)


    def select_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            a = np.array([[x, y]])
            return a / np.linalg.norm(a) 
        else:
            a = self.action(state)
            return a / np.linalg.norm(a)

    def normalize(self, a):
        return keras.utils.normalize(a, axis=1)


