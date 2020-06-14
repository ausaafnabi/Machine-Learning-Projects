from collections import deque
import numpy as np
import argparse
import os
import time
import gym
from gym import wrappers, logger

class QAgent():
    def __init__(self,
                observation_space,
                action_space,
                demo=False,
                slippery=False,
                decay=0.99):
        
        self.action_space = action_space
        #number of columns = number of actions
        col = action_space.n
        #number of rows = number of observation
        row = observation_space.n
        #build Q table with x col dim
        self.q_table = np.zeros([row,col])
        #discount factor
        self.gamma = 0.9
        #initially 90% exploration,10% exploitation
        self.epsilon= 0.9
        #iteratively applying decay til 10% exploration/90% exploitation
        self.epsilon_decay = decay
        self.epsilon_min=0.1

        self.learning_rate =0.1
        #file where ! Table is saved on/restored from
        if slippery:
            self.filename= 'q-frozenlake-slippery.npy'
        else:
            self.filename = 'q-frozenlake.npy'

        self.demo =demo
        if demo:
            self.epsilon=0

    def act(self,state,is_explore=False):
        if is_explore or np.random.rand() <self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.q_table[state])

    #TD[0] learning (generalized Q-Learning) with learning rate
    def update_q_table(self,state,action,reward,next_state):
        q_value = self.gamma *np.argmax(self.q_table[next_state])
        q_value += reward
        q_value -= self.q_table[state,action]
        q_value *=self.learning_rate
        q_value += self.q_table[state,action]
        self.q_table[state,action] = q_value

    #dump Q Table
    def print_q_table(self):
        print(self.q_table)
        print('Epsilon :' self.epsilon)

    #save trained Q Table
    def save_q_table(self):
        np.save(save.filename,self.q_table)

    #load trained Q Table
    def load_q_table(self):
        self.q_table = np.load(sef.filename)

    #adjust epsilon
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsion_decay

if __name__ = '__main__':
    # loop for the specified number of episode
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            # determine the agent's action given state
            action = agent.act(state, is_explore=args.explore)
            # get observable data
            next_state, reward, done, _ = env.step(action)
            # clear the screen before rendering the environment
            os.system('clear')
            # render the environment for human debugging
            env.render()
            # training of Q Table
            if done:
                # update exploration-exploitation ratio
                # reward > 0 only when Goal is reached
                # otherwise, it is a Hole
                if reward > 0:
                    wins += 1
            
            if not args.demo:
                agent.update_q_table(state, action, reward, next_state)
                agent.update_epsilon()
            state = next_state
            percent_wins = 100.0 * wins / (episode + 1)
            print("-------%0.2f%% Goals in %d Episodes---------" % (percent_wins, episode))
            if done:
                time.sleep(5 * delay)
            else:
                time.sleep(delay)
