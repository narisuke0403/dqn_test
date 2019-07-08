import tkinter
import time
import copy
import threading

import numpy as np
from tqdm import tqdm

from stage import Stage, WholeStage
from agent import Agent
import test


def training(stage, player, n_epochs=1000):

    # GUIの設定
    root = tkinter.Tk()
    root.title("")
    root.geometry("400x400")
    canvas = tkinter.Canvas(root, width=400, height=400)
    canvas.place(x=0, y=0)

    test.draw_stage(canvas, stage)

    a = 100
    b = player.replay_memory_size
    c = 100
    first = True
    for _ in tqdm(range(n_epochs)):
        def one_step():
            step = 0
            stage.reset()

            state_t_1, reward_t, terminal = stage.observe()
            # trainging
            try:
                while not terminal:
                    state_t = state_t_1
                    action_t = player.select_action(
                        state_t, player.exploration)
                    stage.execute_action(action_t, step)
                    step += 1
                    state_t_1, reward_t, terminal = stage.observe()

                    if terminal and reward_t == 1:
                        goal_count += 1

                    player.store_experience(
                        state_t, action_t, reward_t, state_t_1, terminal)
            except KeyboardInterrupt:
                player.reward_model.save(
                    "reward_model.h5", include_optimizer=True)
                player.action_model.save(
                    "action_model.h5", include_optimizer=True)
            #c = -(100 / 2) * np.log(len(player.D) / b) + 10  # NOQA
        goal_count = 0
        for _ in range(int(c)):
            step = 0
            stage.reset()

            state_t_1, reward_t, terminal = stage.observe()
            for x in stage.start_position:
                stage.player_pos = x
                # trainging
                try:
                    while not terminal:
                        state_t = state_t_1
                        action_t = player.select_action(
                            state_t, player.exploration)
                        stage.execute_action(action_t, step)
                        step += 1
                        state_t_1, reward_t, terminal = stage.observe()

                        if terminal and reward_t == 1:
                            goal_count += 1

                        player.store_experience(
                            state_t, action_t, reward_t, state_t_1, terminal)
                except KeyboardInterrupt:
                    player.reward_model.save(
                        "reward_model.h5", include_optimizer=True)
                    player.action_model.save(
                        "action_model.h5", include_optimizer=True)
            #c = -(100 / 2) * np.log(len(player.D) / b) + 10  # NOQA
            """
            thread_1 = threading.Thread(target=one_step)
            thread_2 = threading.Thread(target=one_step)

            thread_1.start()
            thread_2.start()
            """
        player.experience_replay()
        player.good_action_replay()

        # stage.reset()

        canvas.delete("object")
        test.draw_oval(canvas, np.array([5, 5]), "black")
        test.draw_all_line(canvas, stage, player)
        canvas.pack()
        canvas.update()

        print(goal_count)
    root.mainloop()

    player.reward_model.save("reward_model.h5", include_optimizer=True)
    player.action_model.save("action_model.h5", include_optimizer=True)


def profiling_test(stage, player):
    data_num = 0
    for _ in tqdm(range(10)):
        for _ in range(50):
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


def profiling():
    import line_profiler

    stage = WholeStage()
    player = Agent()

    pr = line_profiler.LineProfiler()
    pr.add_function(player.experience_replay)
    pr.enable()

    profiling_test(stage, player)

    pr.disable()

    pr.print_stats()


def multiprocess_training():
    pass


if __name__ == "__main__":

    stage = Stage()
    player = Agent()
    """
    profiling_test(stage, player)
    """

    training(stage, player, 10000)
