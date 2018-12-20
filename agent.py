from collections import deque

import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation


class Agent:
    def __init__(self):
        self.minibatch_size = 128
        self.replay_memory_size = 10000
        self.learning_rate = 0.001
        self.discount_factor = 0.9
        self.exploration = 0.1

        # replay memoly
        self.D = deque(maxlen=self.replay_memory_size)

        # action_model
        self.init_action_model()

        # reward_model
        self.init_reward_model()

        # variables
        self.current_loss = 0.0
    
    def init_action_model(self):
        self.action_model = Sequential()
        self.action_model.add(Dense(24,input_shape=(4,)))
        self.action_model.add(Activation("relu"))
        self.action_model.add(Dense(24))
        self.action_model.add(Activation("relu"))
        self.action_model.add(Dense(2))
        self.action_model.compile(
            loss="mean_squared_error",
            optimizer="adam",
        )
    
    def init_reward_model(self):
        self.reward_model = Sequential()
        self.reward_model.add(Dense(24,input_shape=(6,), kernel_initializer=keras.initializers.Zeros()))
        self.reward_model.add(Activation("relu"))
        self.reward_model.add(Dense(24, kernel_initializer=keras.initializers.Zeros()))
        self.reward_model.add(Activation("relu"))
        self.reward_model.add(Dense(1))
        self.reward_model.compile(
            loss="mean_squared_error",
            optimizer="rmsprop",
        )
    
    def reward(self, state):
        return self.reward_model.predict(state)
    
    def action(self, state):
        return self.action_model.predict(state)

    def store_experience(self, state, action, reward, state_1, action_1,terminal):
        self.D.append((state, action, reward, state_1, action_1,terminal))
    
    def experience_replay(self):
        action_state_minibatch = []
        action_y_minibatch = []
        reward_state_minibatch = []
        reward_y_minibatch = []

        minibatch_size = min(len(self.D), self.minibatch_size)
        minibatch_indexes = np.random.randint(0, len(self.D), minibatch_size)
        
        for j in minibatch_indexes:
            state_j, action_j, reward_j, state_j_1, action_j_1 ,terminal = self.D[j]

            reward_state_j = np.hstack((state_j, action_j))
            reward_state_j_1 = np.hstack((state_j_1, action_j_1))
            #y_j_now = self.reward_model.predict(reward_state_j)
            if terminal:
                y_j = reward_j
            else:
                # reward_j + gamma * max_action' Q(state', action')
                y_j = reward_j + self.discount_factor * self.reward_model.predict(reward_state_j_1)  # NOQA
            y_j = np.clip(y_j, -1.0, 1.0)
            #print("now state reward:{}".format(y_j_now))
            #print("update reward:{}".format(y_j))

            # make memory
            # action memory
            if y_j > 0:
                if action_state_minibatch == []:
                    action_state_minibatch = state_j
                else:
                    action_state_minibatch = np.vstack((action_state_minibatch, state_j))
                if action_y_minibatch == []:
                    action_y_minibatch = action_j
                else:
                    action_y_minibatch = np.vstack((action_y_minibatch, action_j))

            # reward memory
            if reward_state_minibatch == []:
                reward_state_minibatch = reward_state_j
            else:
                reward_state_minibatch = np.vstack((reward_state_minibatch, reward_state_j))
            if reward_y_minibatch == []:
                reward_y_minibatch = y_j
            else:
                reward_y_minibatch = np.vstack((reward_y_minibatch, y_j))

        self.reward_model.fit(reward_state_minibatch, reward_y_minibatch, epochs=10)
        if action_state_minibatch != []:
            print("action network fitting")
            self.action_model.fit(action_state_minibatch, action_y_minibatch, epochs=10)
    
    def select_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            print("geedy")
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            a = np.array([[x, y]])
            return (a / np.linalg.norm(a))
        else:
            print("useing network")
            a = self.action(state)
            return (a / np.linalg.norm(a)) 