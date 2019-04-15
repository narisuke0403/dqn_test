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

def draw_stage(canvas, size, whole=None):
    if whole != None:
        for i in whole:
            canvas.create_rectangle(i[0] * scale, i[1] * scale, i[2] * scale, i[3] * scale)

# agent will go to goal
def draw_simple():
    field = stage.Stage()
    player = Agent()
    
    # load model file
    try:
        player.action_model = keras.models.load_model("action_model.h5")
        player.reward_model = keras.models.load_model("reward_model.h5")
    except:
        print("cannot find file")

    # initialize field
    field.reset()
    step = 0
    
    # initialize TKinter
    root = tkinter.Tk()
    root.geometry("400x600")
    canvas = tkinter.Canvas(root, width=400, height=600)
    draw_stage(canvas, field.MAP[0]) 
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
        canvas.create_text(200, 450, text="step = {}".format(step), tag="object")
        canvas.create_text(200, 500, text="action = {}".format(action[0]), tag="object")
        canvas.create_text(200, 550, text="reward = {}".format(reward[0]), tag="object")

        canvas.pack()
        canvas.update()
    
    root.mainloop()

def draw_all_line():
    field = stage.Stage()
    player = Agent()

    # initialize field
    field.reset()

    # initialize TKinter
    root = tkinter.Tk()
    root.geometry("400x400")
    canvas = tkinter.Canvas(root, width=400, height=400)
    draw_stage(canvas, field.MAP[0])
    draw_oval(canvas, field.goal[0], "black")


    # load model file
    try:
        player.action_model = keras.models.load_model("action_model.h5")
        player.reward_model = keras.models.load_model("reward_model.h5")
    except:
        print("cannot find file")
    
    for x in range(1,10):
        for y in range(1,10):
            point = np.array([[x,y]])
            state = np.hstack((point, field.goal))
            action = player.select_action(state, 0)
            draw_line(canvas, point[0], action[0])
    canvas.pack()
    root.mainloop()
            

if __name__ == "__main__":
    draw_simple()
    draw_all_line()
