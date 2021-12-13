import numpy as np
import gym
env = gym.make("Taxi-v3").env
env.reset() # reset environment to a new, random state
env.render()
q_table = np.zeros([env.observation_space.n, env.action_space.n])
"""Training the agent"""
import random
from IPython.display import clear_output
# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1
# For plotting metrics
all_epochs = []
all_penalties = []
training_time=50001
#training_time=10001
print("q_table[128] : ")
print(q_table[128])
print("q_table[228] : ")
print(q_table[228])
print("q_table[328] : ")
print(q_table[328])
for i in range(1, training_time):
    state = env.reset()
    epochs, penalties, reward, = 0, 0, 0
    done = False
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values
        next_state, reward, done, info = env.step(action) 
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value
        if reward == -10:
            penalties += 1
        state = next_state
        epochs += 1
    if i % 1000 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")
print("Training finished.\n")
print("q_table[128] : ")
print(q_table[128])
print("q_table[228] : ")
print(q_table[228])
print("q_table[328] : ")
print(q_table[328])
total_epochs, total_penalties = 0, 0
episodes = 100
for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False
    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)
        if reward == -10:
            penalties += 1
        epochs += 1
    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")

