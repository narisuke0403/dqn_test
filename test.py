import csv
import tkinter
import time

import numpy as np
import keras

import stage
from agent import Agent

margin = 0
scale = 40


def draw_line(canvas, point, direction):
    canvas.create_line(point[0] * scale + margin, point[1] * scale + margin, point[0] *
                       scale + margin + direction[0] * scale, point[1] * scale + margin + direction[1] * scale, tag="object")


def draw_oval(canvas, point, color):
    canvas.create_oval(point[0] * scale + margin - scale/2, point[1] * scale + margin -
                       scale/2, point[0] * scale + margin + scale/2, point[1] * scale + margin +
                       scale/2, tag="object", fill=color)


def draw_stage(canvas, field):
    for i in field.whole:
        if field.whole.size != 0:
            canvas.create_rectangle(
                i[0] * scale, i[1] * scale, i[2] * scale, i[3] * scale)


# agent will go to goal
def draw_simple(field, goal):
    player = Agent()

    # load model file
    try:
        player.action_model = keras.models.load_model("action_model.h5")
        player.reward_model = keras.models.load_model("reward_model.h5")
    except:
        print("cannot find file")

    # initialize field
    field.reset()

    # define player's position and goal position
    #field.goal = goal
    #field.player_pos = np.array([[0.5, 0.5]])
    step = 0

    # initialize TKinter
    root = tkinter.Tk()
    root.geometry("400x600")
    canvas = tkinter.Canvas(root, width=400, height=600)
    draw_stage(canvas, field)
    draw_oval(canvas, field.start[0], "red")
    draw_oval(canvas, field.goal[0], "black")

    while field.terminal == False:
        time.sleep(1)
        step += 1
        state, _, _ = field.observe()
        action = player.select_action(state, 0)
        state_reward = np.hstack((state, action))
        reward = player.reward(state_reward)
        field.execute_action(action, step)

        # draw stage
        canvas.delete("object")
        draw_oval(canvas, field.goal[0], "black")
        draw_oval(canvas, field.player_pos[0], "red")

        # draw detail
        canvas.create_text(
            200, 450, text="step = {}".format(step), tag="object")
        canvas.create_text(
            200, 500, text="action = {}".format(action[0]), tag="object")
        canvas.create_text(
            200, 550, text="reward = {}".format(reward[0]), tag="object")

        canvas.pack()
        canvas.update()

    root.mainloop()


def draw_all_line(canvas, field, player):
    for x in range(1, 10):
        for y in range(1, 10):
            point = np.array([[x, y]])
            if field.check_MAP(point):
                state = np.hstack((point, field.goal))
                action = player.select_action(state, 0)
                draw_line(canvas, point[0], action[0])


def all_line(goal, field, player):

    def click(event):
        canvas.delete("object")
        goal = np.array([[event.x / 40.0, event.y / 40.0]])
        draw_oval(canvas, goal[0], "black")
        field.goal = goal
        draw_all_line(canvas, field, player)
        canvas.pack()
        canvas.update()

    # initialize field
    field.reset()
    #field.goal = goal

    # initialize TKinter
    root = tkinter.Tk()
    root.geometry("400x400")
    canvas = tkinter.Canvas(root, width=400, height=400)
    canvas.bind("<Button-1>", click)
    draw_stage(canvas, field)

    # load model file
    try:
        player.action_model = keras.models.load_model("action_model.h5")
        player.reward_model = keras.models.load_model("reward_model.h5")
    except:
        print("cannot find file")

    draw_oval(canvas, field.goal[0], "black")
    draw_all_line(canvas, field, player)
    canvas.pack()
    root.mainloop()


def simple(canvas, field, player):
    # time.sleep(0.005)

    draw_stage(canvas, field)
    draw_all_line(canvas, field, player)
    draw_oval(canvas, field.player_pos[0], "black")
    draw_oval(canvas, field.goal[0], "red")
    canvas.pack()
    canvas.update()


if __name__ == "__main__":
    field = stage.Stage()
    player = Agent()
    goal = np.array([[5, 5]])

    draw_simple(field, goal)
    all_line(goal, field, player)
