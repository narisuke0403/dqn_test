import csv
import tkinter
import time

import numpy as np
import keras

from stage import Stage
from agent import Agent


def draw(canvas, env, agent, number=0, step=0):
    time.sleep(0.5)
    
    # draw info
    canvas.delete("player")
    canvas.create_rectangle(100, 100, 500, 500)

    # is there whole
    if env.iswhole:
        canvas.create_rectangle(
            100 + env.whole[0][0]*40,
            100 + env.whole[0][2]*40,
            100 + env.whole[0][1]*40, 
            100 + env.whole[0][3]*40
            )
        
    canvas.create_oval(env.start[0][0] * 40.0 + 100, env.start[0][1] * 40.0 + 100, env.start[0][0] * 40.0 + 50 + 100, env.start[0][1] * 40.0 + 50 + 100, tag="start")
    canvas.create_oval(env.goal[0][0] * 40.0 + 100, env.goal[0][1] * 40.0 + 100,env.goal[0][0] * 40.0 + 50 + 100, env.goal[0][1] * 40.0 + 50 + 100, tag="goal")
    canvas.create_oval(env.player_pos[0][0] * 40.0 + 100, env.player_pos[0][1] * 40.0 + 100, env.player_pos[0][0] * 40.0 + 10 + 100, env.player_pos[0][1] * 40.0 + 10 + 100, tag="player")
    canvas.create_oval(env.player_pos[0][0] * 40.0 + 100 -10 , env.player_pos[0][1] * 40.0 + 100 - 10, env.player_pos[0][0] * 40.0 + 10 + 100 + 10, env.player_pos[0][1] * 40.0 + 10 + 100 + 10, tag="player")
    # debug score
    state = np.hstack((env.player_pos, env.goal))
    state_action = np.hstack((state, agent.action(state_n)))
    reward_action = agent.reward_model.predict(state_action_n)
    for x in range(-1,1):
        for y in range(-1,1):
            state = np.hstack((env.player_pos+np.array((x, y)), env.goal))
            state_action = np.hstack((state, agent.action(state_n)))
            reward_action = agent.reward_model.predict(state_action_n)
    canvas.delete("line")
    canvas.create_text(250, 35, text="pos : {}".format(state_action), tag="line")
    canvas.create_text(250, 50, text="taining : {}".format(number), tag="line")
    canvas.create_text(250, 65, text="step : {}".format(step), tag="line")
    canvas.create_text(250, 90, text="reward action : {}".format(reward_action), tag="line")
    
    canvas.pack()
    canvas.update()


def _from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % rgb

def draw_all(canvas, env,agent, number=0):
    canvas.create_rectangle(100, 100, 500, 500)
    _, goal = env.set_start_goal()
    xs = np.arange(0.0, 10.5, 0.5)
    ys = np.arange(0.0, 10.5, 0.5)
    canvas.delete("line")
    if env.iswhole:
        canvas.create_rectangle(100 + env.whole[0][0]*40,100 + env.whole[0][1]*40,100 + env.whole[0][2]*40,100 + env.whole[0][3]*40,tag="line")
    canvas.create_oval(goal[0][0] * 40.0 + 100-25, goal[0][1] * 40.0 + 100-25, goal[0][0] * 40.0 + 25 + 100, goal[0][1] * 40.0 + 25 + 100, tag="line")
    canvas.create_text(250, 50, text="taining : {}".format(number), tag="line")
    canvas.create_text(250, 80, text="number of action experience : {}".format(len(agent.good_action_experience)), tag="line")
    for x in xs:
        for y in ys:
            a = np.array([[x, y]])
            if env.check_MAP(a):
                s = np.hstack((a, goal))
                ac = agent.action_model.predict(s)
                d = np.linalg.norm(ac)
                ac = ac / d #if d > 1 else ac
                canvas.create_line(100 + x * 40, 100 + y * 40, 100 + (x + ac[0][0]) * 40, 100 + (y + ac[0][1]) * 40, tag="line")
                canvas.pack()
    canvas.update()


def move_check(canvas, env, agent, number):
    
    env.reset()

     # テスト環境設定
    env.start = np.array([[1.0, 1.0]])
    env.goal = np.array([[8.0, 8.0]])
    env.player_pos = np.copy(env.start)

    state_t_1, _, terminal, _ = env.observe()
    step = 0
    while not terminal:
        state_t = state_t_1
        state_t_n = agent.normalize(state_t)
        action_t = agent.select_action(state_t_n, 0)
        env.execute_action(action_t, step)
        step += 1
        draw(canvas, env, agent, number, step)
        state_t_1, _, terminal, _ = env.observe()
    draw(canvas, env, agent, number, step)

def vector_check(env):
    agent = Agent()
    agent.action_model = keras.models.load_model("action_model.h5", compile=False)

    # GUIの設定
    root = tkinter.Tk()
    root.title("")
    root.geometry("600x600")
    canvas = tkinter.Canvas(root, width=600, height=600)
    canvas.place(x=0, y=0)
    draw_all(canvas, env, agent)
    root.mainloop()


if __name__ == "__main__":
    env = Stage()
    agent = Agent()

    # テスト環境設定
    env.start = np.array([[2.0, 2.0]])
    env.goal = np.array([[8.0, 8.0]])
    env.player_pos = np.copy(env.start)

    # 地点のランダム生成
    env.reset()

    # モデルの読み込み
    agent.action_model = keras.models.load_model("action_model.h5", compile=False)
    agent.reward_model = keras.models.load_model("reward_model.h5", compile=False)

    state_t_1, _, terminal = env.observe()
    state = []
    step = 0

    # GUIの設定
    root = tkinter.Tk()
    root.title("")
    root.geometry("600x600")
    canvas = tkinter.Canvas(root, width=600, height=600)
    canvas.place(x=0, y=0)

    """
    while not terminal:
        state_t = state_t_1
        state.append(np.copy(env.player_pos))
        action_t = agent.select_action(state_t, 0)
        env.execute_action(action_t, step)
        step += 1
        draw(canvas, env, agent)
        state_t_1, _, terminal = env.observe()
    draw(canvas, env, agent)
    """
    while True:
        time.sleep(1)
        draw_all(canvas, env, agent)
    root.mainloop()
    
