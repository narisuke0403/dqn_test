import csv
import tkinter
import time

import numpy as np
import keras

from stage import Stage
from agent import Agent


def draw(canvas, env):

    time.sleep(1)
    canvas.create_rectangle(100, 100, 500, 500)
    canvas.create_oval(env.start[0][0] * 40.0 + 100, env.start[0][1] * 40.0 + 100, env.start[0][0] * 40.0 + 50 + 100, env.start[0][1] * 40.0 + 50 + 100, tag="start")
    canvas.create_oval(env.goal[0][0] * 40.0 + 100, env.goal[0][1] * 40.0 + 100,env.goal[0][0] * 40.0 + 50 + 100, env.goal[0][1] * 40.0 + 50 + 100, tag="goal")
    canvas.create_oval(env.player_pos[0][0] * 40.0 + 100, env.player_pos[0][1] * 40.0 + 100,env.player_pos[0][0] * 40.0 + 10 + 100, env.player_pos[0][1] * 40.0 + 10 + 100, tag="player")
    canvas.pack()
    canvas.update()

if __name__ == "__main__":
    env = Stage()
    agent = Agent()

    # テスト環境設定
    env.start = np.array([[2.0, 2.0]])
    env.goal = np.array([[8.0, 8.0]])
    env.player_pos = np.copy(env.start)

    # 地点のランダム生成
    #env.reset()
    
    # モデルの読み込み
    agent.action_model = keras.models.load_model("action_model.h5", compile=False)

    state_t_1, _, terminal, action = env.observe()
    state = []
    step = 0

    # GUIの設定
    root = tkinter.Tk()
    root.title("")
    root.geometry("600x600")
    canvas = tkinter.Canvas(root, width=600, height=600)
    canvas.place(x=0, y=0)

    while not terminal:
        state_t = state_t_1
        state.append(np.copy(env.player_pos))
        action_t = agent.select_action(state_t, 0)
        env.execute_action(action_t, step)
        step += 1
        draw(canvas, env)
        state_t_1, reward_t, terminal, action_t_1 = env.observe()
    root.mainloop()
    print(state)
    

