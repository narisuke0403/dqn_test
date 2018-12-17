from math import sqrt
from collections import deque
import numpy as np
import tensorflow as tf

class Env:

    MAP = np.array([100, 100])

    def __init__(self, agent=object, *args, **kwargs):
        self.agent = agent
        
    def step(self,):
        action = self.agent.action()
        self.agent.next = self.agent.now + action
        if self._check_MAP(self.agent.next):
            self.agent.now = self.agent.next
        else:
            self.agent.alive = False

    def reset(self,):
        self.agent.now = self.agent.START

    def _check_MAP(self,pos):
        if min([pos[0], self.MAP[0]]) < self.MAP[0] and min([pos[1], self.MAP[1]]) < self.MAP[1]:
            if max([pos[0], 0]) > 0 and max([pos[1], 0]) > 0:
                return True
        print("Get out of the Field")
        return False

    def _monitor(self):
        print("Agent.alive:{}".format(self.agent.alive))
        print("self.agent.position:({})".format(self.agent.now))
        print("reward:{}".format(self.agent.reward()))

    def run(self, try_nb, MAX_STEP=100,):
        for _ in range(try_nb):
            sp = 0
            while self.agent.alive:
                sp += 1
                if sp > MAX_STEP:
                    break
                self.step()
                self._monitor()
            self.reset()
                
            
class Agent:
    
    goal = None

    def __init__(self,*args, **kwargs):
        self.START = np.array([10, 10])
        self.now = self.START
        self.next = self.now
        self.alive = True
        self.SPEED = 3.0
        self.decide_goal()

        # for agent training
        self.minibatch_size = 32
        self.replay_memory_size = 1000
        self.learning_rate = 0.001
        self.discount_factor = 0.9
        self.exploration = 0.1

        # replay memory
        self.D = deque(maxlen=self.replay_memory_size)

        # model
        self.init_model()

        # variables
        self.current_loss = 0.0
    
    def action(self):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        a = np.array([x, y])
        return (a / np.linalg.norm(a)) * self.SPEED
    
    def init_Q_model(self):
        pass

    def init_reward_model(self,):
        pass

    def Q_values(self, state):
        return 0

    def select_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            a = np.array([x, y])
            return (a / np.linalg.norm(a)) * self.SPEED
        else:
            a = self.Q_values(state)
            return a
    
    def store_experience(self, state, action, reward, state_1, terminal):
        self.D.append((state, action, reward, state_1, terminal))

    def experience_replay(self):
        state_minibatch = []
        y_minibatch = []

        # sample random minibatch
        minibatch_size = min(len(self.D), self.minibatch_size)
        minibatch_indexes = np.random.randint(0, len(self.D), minibatch_size)

        
    def decide_goal(self):
        self.goal = np.array([80,80])

    def reward(self,):
        return self._dict(self.goal, self.now)

    def _dict(self,x,y):
        d = np.linalg.norm(x-y)
        return d

if __name__ == '__main__':
    agent = Agent()
    env = Env(agent=agent)
    env.run(20)
    
