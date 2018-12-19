import numpy as np

from stage import Stage
from agent import Agent

if __name__ == "__main__":
    env = Stage()
    agnet = Agent()

    # parameters
    n_epochs = 10

    goal = 0

    for i in range(n_epochs):
        step = 0
        env.reset()
        state_t_1, reward_t, terminal, action = env.observe()
        state = []
        while not terminal:
            state_t = state_t_1
            state.append(env.player_pos)
            action_t = agnet.select_action(state_t, agnet.exploration)
            env.execute_action(action_t, step)
            step += 1

            state_t_1, reward_t, terminal, action_t_1 = env.observe()
            
            agnet.store_experience(state_t, action_t, reward_t, state_t_1, action_t_1 ,terminal)

            agnet.experience_replay()
        print(state)