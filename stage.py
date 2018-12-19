import numpy as np

class Stage:
    def __init__(self):
        self.MAP = np.array([[10.0, 10.0]])
        self.START = np.array([[2.0, 2.0]])
        self.player_pos = np.copy(self.START)
        self.GOAL = np.array([[8.0, 8.0]])
        self.action = np.array([[0.0, 0.0]])
        self.MAXSTEP = 100.0
        self.reset()
    
    def reset(self):
        self.player_pos = np.copy(self.START)
        self.reward = 0
        self.terminal = False
    
    def update(self, action, step):
        self.action = action
        self.player_pos += self.action
        self.reward = 0
        self.terminal = False
        if self._dict(self.player_pos, self.GOAL) < 1.0:
            self.reward = 1
            self.terminal = True
        elif step > self.MAXSTEP or self._check_MAP(self.player_pos) == False:
            self.reward = -1
            self.terminal = True

    def observe(self):
        state = np.hstack((self.player_pos, self.GOAL))
        return state, self.reward, self.terminal, self.action

    def execute_action(self, action, step):
        self.update(action, step)

    def _dict(self, x, y):
        d = np.linalg.norm(x-y)
        return d
    
    def _check_MAP(self,pos):
        if min([pos[0][0], self.MAP[0][0]]) < self.MAP[0][0] and min([pos[0][1], self.MAP[0][1]]) < self.MAP[0][1]:
            if max([pos[0][0], 0]) > 0 and max([pos[0][1], 0]) > 0:
                return True
        print("Get out of the Field")
        return False