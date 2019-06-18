import tkinter
import time
import copy
from multiprocessing import Pool
import multiprocessing

import numpy as np
from tqdm import tqdm

from stage import Stage, WholeStage
from agent import Agent
import test


def training(stage, player):

    # GUIの設定
    root = tkinter.Tk()
    root.title("")
    root.geometry("400x400")
    canvas = tkinter.Canvas(root, width=400, height=400)
    canvas.place(x=0, y=0)

    test.draw_stage(canvas, stage)

    n_epochs = 100000
    data_num = 0
    for _ in tqdm(range(n_epochs)):
        for _ in range(10):
            data_num += 1
            step = 0
            stage.reset()

            state_t_1, reward_t, terminal = stage.observe()

            # trainging
            step = 0
            stage.reset()

            state_t_1, reward_t, terminal = stage.observe()
            while not terminal:
                state_t = state_t_1
                action_t = player.select_action(state_t, player.exploration)
                stage.execute_action(action_t, step)
                step += 1

                state_t_1, reward_t, terminal = stage.observe()

                player.store_experience(
                    state_t, action_t, reward_t, state_t_1, terminal)
        player.experience_replay()
        player.good_action_replay()

        # stage.reset()
        canvas.delete("object")
        test.draw_oval(canvas, stage.goal[0], "black")
        test.draw_all_line(canvas, stage, player)
        canvas.pack()
        canvas.update()
    root.mainloop()
    player.reward_model.save("reward_model.h5", include_optimizer=True)
    player.action_model.save("action_model.h5", include_optimizer=True)


if __name__ == "__main__":
    stage = Stage()
    player = Agent()
    training(stage, player)
