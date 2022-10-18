import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from point_env import *
import pygame

def get_init_hidden():
    pass

goal = semi_circle_goal_sampler()
logger.debug(goal)
rnn_policy = None

env = SparsePointEnv(goal_radius=0.5, max_episode_steps=100, goal_sampler='semi-circle', is_render=True)

for e in range(10):
    done = False
    obs = env.reset()
    hidden = get_init_hidden()
    while not done:
        action, hidden = rnn_policy(obs, hidden)
        next_obs, reward, done, info = env.step(action)

    
    