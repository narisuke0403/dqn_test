import numpy as np


class Stage:
    whole = np.array([[]])

    def __init__(self):
        self.MAP = np.array([[10.0, 10.0]])
        self.MAXSTEP = 20
        self.reset()
        self.goal_position = []
        self.start_position = []
        self.set_goal()
        self.set_start()

    def set_start_goal(self):
        self.goal = np.array(
            [[np.random.rand() * 8.0 + 1, np.random.rand() * 8.0 + 1]])
        self.start = np.array(
            [[np.random.rand() * 8.0 + 1, np.random.rand() * 8.0 + 1]])
        while self.check_MAP(self.goal) == False:
            self.goal = np.array(
                [[np.random.rand() * 8.0 + 1, np.random.rand() * 8.0 + 1]])
        while self.check_MAP(self.start) == False:
            self.start = np.array(
                [[np.random.rand() * 8.0 + 1, np.random.rand() * 8.0 + 1]])
        return self.start, self.goal

    def set_goal(self):
        for x in range(1, 10, 8):
            for y in range(1, 10, 8):
                self.goal_position.append(np.array([[x, y]]))

    def set_start(self):
        for x in np.arange(1.0, 10.0, 8.0):
            for y in np.arange(1.0, 10.0, 8.0):
                self.start_position.append(np.array([[x, y]]))

    def reset(self):
        self.reward = 0
        self.terminal = False
        self.set_start_goal()
        self.player_pos = np.copy(self.start)
        self.action = np.array([[0.0, 0.0]])

    def update(self, action, step):
        self.action = action
        self.before_player_pos = np.copy(self.player_pos)
        self.player_pos += self.action
        self.reward = 0
        self.terminal = False
        if self._distance(self.goal[0][0], self.goal[0][1], self.before_player_pos[0][0], self.before_player_pos[0][1], self.player_pos[0][0], self.player_pos[0][1]) <= 1.0:
            # print("GOAL")
            self.reward = 1
            self.terminal = True
        elif step > self.MAXSTEP or self.check_MAP(self.player_pos) == False or self._judge_inside(self.whole, self.before_player_pos):
            self.reward = -1
            self.terminal = True

    def observe(self):
        state = np.hstack((self.player_pos, self.goal))
        return state, self.reward, self.terminal

    def execute_action(self, action, step):
        self.update(action, step)

    def check_MAP(self, pos):
        if min([pos[0][0], self.MAP[0][0]]) < self.MAP[0][0] and min([pos[0][1], self.MAP[0][1]]) < self.MAP[0][1]:
            if max([pos[0][0], 0]) > 0 and max([pos[0][1], 0]) > 0:
                return True
        # print("Get out of the Field")
        return False

    def _distance(self, px, py, x1, y1, x2, y2):
        a = x2 - x1
        b = y2 - y1
        a2 = a * a
        b2 = b * b
        r2 = a2 + b2
        tt = -(a * (x1 - px) + b * (y1 - py))
        if tt < 0:
            return (x1 - px) * (x1 - px) + (y1 - py) * (y1 - py)
        if tt > np.sqrt(r2):
            return (x2 - px) * (x2 - px) + (y2 - py) * (y2 - py)

        f1 = a * (y1 - py) - b * (x1 - px)
        return f1 * f1 / r2

    def _judge_inside(self, wholes, before_player_pos):
        if wholes.size != 0:
            for whole in wholes:
                points = ((whole[0], whole[1]), (whole[2], whole[1]),
                          (whole[2], whole[3]), (whole[0], whole[3]))
                for point in points:
                    x1 = self.player_pos[0][0]
                    y1 = before_player_pos[0][1]
                    x2 = self.player_pos[0][0]
                    y2 = self.player_pos[0][1]
                    x3 = before_player_pos[0][0]
                    y3 = before_player_pos[0][1]
                    x4 = point[0]
                    y4 = point[1]
                    A = x2-x1
                    B = x3-x1
                    C = y2-y1
                    D = y3-y1
                    if ((x4*C - y4*A)/(C*B-A*D) <= 1.0) and ((-x4*D + y4*B)/(C*B-A*D) <= 1.0):
                        return True
            return False


class WholeStage(Stage):
    whole = np.array([[4.5, 4.5, 6.5, 6.5]])

    def __init__(self, *args, **kwargs):
        super().__init__()

        # make whole x1,y1,x2,y2, length_x=x2-x1, length_y=y2-y1
        self.iswhole = True

    def check_MAP(self, pos):
        if pos[0][0] > self.whole[0][0] and pos[0][0] < self.whole[0][2] and pos[0][1] > self.whole[0][1] and pos[0][1] < self.whole[0][3]:
            return False
        if min([pos[0][0], self.MAP[0][0]]) < self.MAP[0][0] and min([pos[0][1], self.MAP[0][1]]) < self.MAP[0][1]:
            if max([pos[0][0], 0]) > 0 and max([pos[0][1], 0]) > 0:
                return True
        return False
