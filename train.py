import numpy as np

from stage import Stage
from agent import Agent

if __name__ == "__main__":
    env = Stage()
    agent = Agent()

    # parameters
    n_epochs = 10000

    goal = 0

    for i in range(n_epochs):
        step = 0
        env.reset()
        state_t_1, reward_t, terminal, action = env.observe()
        state = []
        while not terminal:
            state_t = state_t_1
            state.append(np.copy(env.player_pos))
            action_t = agent.select_action(state_t, agent.exploration)
            env.execute_action(action_t, step)
            step += 1

            state_t_1, reward_t, terminal, action_t_1 = env.observe()
            
            agent.store_experience(
                state_t, action_t, reward_t, state_t_1, action_t_1, terminal)

            agent.experience_replay()
        print(state)
    agent.reward_model.save("reward_model.h5", include_optimizer=False)
    agent.action_model.save("action_model.h5", include_optimizer=False)
