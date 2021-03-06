from collections import deque
import copy
import _pickle as cPickle

import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras import backend as K
from tensorflow import math as T

from tensorflow.python.client import device_lib


class Agent:
    def __init__(self):
        self.replay_memory_size = 1000
        self.discount_factor = 0.9
        self.alpha = 0.5
        self.exploration = 0.25

        # replay memoly
        self.D = deque(maxlen=self.replay_memory_size)
        self.good_action_experience = deque(maxlen=self.replay_memory_size)

        # action_model
        self.init_action_model()

        # reward_model
        self.init_reward_model()
        # self.before_reward_model = copy.deepcopy(self.reward_model)
        self.before_reward_model = cPickle.loads(
            cPickle.dumps(self.reward_model, -1))

        # variables
        self.current_loss = 0.0

        # self.load_model()

    def init_action_model(self):
        self.action_model = Sequential()
        self.action_model.add(Dense(100, input_shape=(
            10,), activation="relu"))
        self.action_model.add(BatchNormalization())
        self.action_model.add(Dropout(0.2))
        self.action_model.add(
            Dense(200, activation="relu"))
        self.action_model.add(BatchNormalization())
        self.action_model.add(Dropout(0.5))
        self.action_model.add(
            Dense(200, activation="relu"))
        self.action_model.add(BatchNormalization())
        self.action_model.add(Dropout(0.5))
        self.action_model.add(
            Dense(100, activation="relu"))
        self.action_model.add(BatchNormalization())
        self.action_model.add(Dropout(0.5))
        self.action_model.add(Dense(2))
        self.action_model.compile(
            loss="cosine_proximity",
            optimizer="adam",
        )

    def init_reward_model(self):
        self.reward_model = Sequential()
        self.reward_model.add(Dense(100, input_shape=(
            14,), kernel_initializer=keras.initializers.he_normal(), bias_initializer='zeros', activation="relu"))
        self.reward_model.add(BatchNormalization())
        self.reward_model.add(Dropout(0.2))
        self.reward_model.add(
            Dense(200, kernel_initializer=keras.initializers.he_normal(), bias_initializer='zeros', activation="relu"))
        self.reward_model.add(BatchNormalization())
        self.reward_model.add(Dropout(0.5))
        self.reward_model.add(
            Dense(300, kernel_initializer=keras.initializers.he_normal(), bias_initializer='zeros', activation="relu"))
        self.reward_model.add(BatchNormalization())
        self.reward_model.add(Dropout(0.5))
        self.reward_model.add(
            Dense(200, kernel_initializer=keras.initializers.he_normal(), bias_initializer='zeros', activation="relu"))
        self.reward_model.add(BatchNormalization())
        self.reward_model.add(Dropout(0.5))
        self.reward_model.add(
            Dense(100, kernel_initializer=keras.initializers.he_normal(), bias_initializer='zeros', activation="relu"))
        self.reward_model.add(BatchNormalization())
        self.reward_model.add(Dropout(0.5))
        self.reward_model.add(
            Dense(1, kernel_initializer=keras.initializers.he_normal(), bias_initializer='zeros'))
        self.reward_model.compile(
            loss="mse",
            optimizer="adam",
        )

    def load_model(self):
        self.reward_model = keras.models.load_model("reward_model.h5")
        self.action_model = keras.models.load_model("action_model.h5")

    def reward(self, state):
        with tf.device("/device:CPU:0"):
            state = self.make_input(state)
            return self.reward_model.predict(state)

    def action(self, state):
        with tf.device("/device:CPU:0"):
            state = self.make_input(state)
            return self.action_model.predict(state)

    def store_experience(self, state, action, reward, state_1, terminal, first):
        self.D.append((state, action, reward, state_1, terminal))

    def select_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            a = np.array([[x, y]])
            return a / np.linalg.norm(a)
        else:
            a = self.action(state)
            return a / np.linalg.norm(a)

    def experience_replay(self, first):
        reward_state_minibatch = []
        reward_y_minibatch = []
        minibatch_indexes = np.random.randint(0, len(self.D), len(self.D))
        for j in minibatch_indexes:
            # get data
            state_j, action_j, reward_j, state_j_1, terminal = self.D[j]

            # get predict data
            reward_state_j = np.hstack((state_j, action_j))
            action_1 = self.select_action(state_j_1, -1)
            reward_state_j_1 = np.hstack((state_j_1, action_1))

            if first:
                y_j_now = reward_j
            else:
                y_j_now = self.reward(reward_state_j)
                y_j_now = np.clip(y_j_now, -1, 1)

            reward_state_j_1 = self.make_input(reward_state_j_1)
            y_j_next = self.before_reward_model.predict(reward_state_j_1)
            y_j_next = np.clip(y_j_next, -1, 1)

            # calculate reward
            if terminal:
                y_j = reward_j
                if y_j > 0:
                    self.good_action_experience.append(
                        self.D[j])
            else:
                if first:
                    y_j = y_j_now
                else:
                    y_j = (1 - self.alpha) * y_j_now[0] + self.alpha * (reward_j + self.discount_factor * y_j_next[0] - y_j_now[0])  # NOQA
                    y_j = np.clip(y_j, -1.0, 1.0)

                if (y_j > 0) and (y_j > y_j_now):
                    self.good_action_experience.append(
                        self.D[j])

            # make reward memory batch
            if reward_state_minibatch == []:
                reward_state_minibatch = reward_state_j
            else:
                reward_state_minibatch = np.vstack(
                    (reward_state_minibatch, reward_state_j))
            if reward_y_minibatch == []:
                reward_y_minibatch.append(y_j)
            else:
                reward_y_minibatch.append(y_j)
        reward_y_minibatch = np.array(reward_y_minibatch)

        reward_state_minibatch = self.make_input(reward_state_minibatch)
        reward_state_minibatch = self.make_trainig_data(reward_state_minibatch)
        self.before_reward_model.set_weights(self.reward_model.get_weights())
        self.reward_model.fit(reward_state_minibatch,
                              reward_y_minibatch, epochs=200, verbose=0, batch_size=256)
        # print(self.reward_model.get_weights())
        self.D.clear()

    def good_action_replay(self):
        if len(self.good_action_experience) != 0:
            action_state_minibatch = []
            action_y_minibatch = []
            minibatch_size = min(
                len(self.good_action_experience), len(self.good_action_experience))
            minibatch_indexes = np.random.randint(
                0, len(self.good_action_experience), minibatch_size)

            for j in minibatch_indexes:
                state_j, action_j, _, _, _, = self.good_action_experience[j]
                if action_state_minibatch == []:
                    action_state_minibatch = state_j
                else:
                    action_state_minibatch = np.vstack(
                        (action_state_minibatch, state_j))
                if action_y_minibatch == []:
                    action_y_minibatch = action_j
                else:
                    action_y_minibatch = np.vstack(
                        (action_y_minibatch, action_j))

            action_state_minibatch = self.make_input(action_state_minibatch)
            action_state_minibatch = self.make_trainig_data(
                action_state_minibatch)
            self.action_model.fit(action_state_minibatch,
                                  action_y_minibatch, epochs=200, verbose=0, batch_size=256)
            # self.good_action_experience.clear()

    def make_input(self, A):
        factorial_a = A ** 2
        vec_s_g = A[:, 0:2] - A[:, 2:4]
        n = np.hstack((A, factorial_a))
        return np.hstack((n, vec_s_g))

    def make_trainig_data(self, A):
        mean = A.mean(axis=0)
        std = A.std(axis=0)

        A = (A - mean) / std

        return A

    def _make_random_data(self):
        random_data = []
        random_data_y = []
        for _ in range(100):
            random_data.append(np.random.uniform(0.5, 9.5, 4))
            random_data_y.append(np.random.uniform(-1, 1, 2))
        random_data = self.make_input(np.array(random_data))
        random_data_y = np.array(random_data_y)
        self.action_model.fit(random_data, random_data_y, verbose=1)
        print(self.action_model.get_weights())


if __name__ == "__main__":
    agent = Agent()
    agent._make_random_data()
