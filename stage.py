import numpy as np

class Stage:
    whole = np.array([[]])
    def __init__(self):
        self.MAP = np.array([[10.0, 10.0]])
        self.MAXSTEP = 10
        self.reset()
        self.goal_position = []
        self.set_goal()
        
    
    def set_start_goal(self):
        self.goal = np.array([[np.random.rand() * 8.0 + 1, np.random.rand() * 8.0 + 1]])
        self.start = np.array([[np.random.rand() * 8.0 + 1, np.random.rand() * 8.0 + 1]])
        while self.check_MAP(self.goal) == False:
            self.goal = np.array([[np.random.rand() * 8.0 + 1, np.random.rand() * 8.0 + 1]])
        while self.check_MAP(self.start) == False:
            self.start = np.array([[np.random.rand() * 8.0 + 1, np.random.rand() * 8.0 + 1]])
        return self.start, self.goal

    def set_goal(self):
        for x in range(1, 10, 2):
            for y in range(1, 10, 2):
                self.goal_position.append(np.array([[x,y]]))

    def reset(self):
        self.reward = 0
        self.terminal = False
        self.set_start_goal()
        self.player_pos = np.copy(self.start)
        self.action = np.array([[0.0, 0.0]])

    def update(self, action, step):
        self.action = action
        self.player_pos += self.action
        self.reward = 0
        self.terminal = False
        if self._dict(self.player_pos, self.goal) < 1.0:
            #print("GOAL")
            self.reward = 1
            self.terminal = True
        elif step > self.MAXSTEP or self.check_MAP(self.player_pos) == False:
            self.reward = -1
            self.terminal = True

    def observe(self):
        state = np.hstack((self.player_pos, self.goal))
        return state, self.reward, self.terminal

    def execute_action(self, action, step):
        self.update(action, step)

    def _dict(self, x, y):
        d = np.linalg.norm(x-y)
        return d
    
    def check_MAP(self,pos):
        if min([pos[0][0], self.MAP[0][0]]) < self.MAP[0][0] and min([pos[0][1], self.MAP[0][1]]) < self.MAP[0][1]:
            if max([pos[0][0], 0]) > 0 and max([pos[0][1], 0]) > 0:
                return True
        # print("Get out of the Field")
        return False

class WholeStage(Stage):
    whole = np.array([[4.5, 4.5,6.5, 6.5]])
    def __init__(self, *args, **kwargs):
        super().__init__()

        # make whole x1,y1,x2,y2, length_x=x2-x1, length_y=y2-y1
        print(self.whole)
        self.iswhole = True
        
    
    def check_MAP(self, pos):
        if pos[0][0] > self.whole[0][0] and pos[0][0] < self.whole[0][2] and pos[0][1] > self.whole[0][1] and pos[0][1] < self.whole[0][3]:
            return False
        if min([pos[0][0], self.MAP[0][0]]) < self.MAP[0][0] and min([pos[0][1], self.MAP[0][1]]) < self.MAP[0][1]:
            if max([pos[0][0], 0]) > 0 and max([pos[0][1], 0]) > 0:
                return True
        return False
