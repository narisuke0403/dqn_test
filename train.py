import tkinter
import time

import numpy as np
from tqdm import tqdm

from stage import Stage, WholeStage
from agent import Agent
import test

def training(stage, player):
    env = stage
    agent = player

    # parameters
    n_epochs = 10000

    """
    # GUIの設定
    root = tkinter.Tk()
    root.title("")
    root.geometry("600x600")
    canvas = tkinter.Canvas(root, width=600, height=600)
    canvas.place(x=0, y=0)
    """
    data_num = 0

    for i in range(n_epochs):
        data_num += 1
        print(data_num)
        step = 0
        env.reset()
        state_t_1, reward_t, terminal = env.observe()
        while not terminal:
            state_t = state_t_1
            action_t = agent.select_action(state_t, agent.exploration)
            env.execute_action(action_t, step)
            step += 1

            state_t_1, reward_t, terminal = env.observe()

            agent.store_experience(
                state_t, action_t, reward_t, state_t_1, terminal)
            if i > 4000:
                agent.experience_replay()
            if len(agent.good_action_experience) > 32:
                agent.good_action_replay()
    agent.reward_model.save("reward_model.h5", include_optimizer=True)
    agent.action_model.save("action_model.h5", include_optimizer=True)

if __name__ == "__main__":
    training(Stage(), Agent())


