import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from point_env import *
import pygame

goal = semi_circle_goal_sampler()
logger.debug(goal)
# env = PointEnv(max_episode_steps=100, is_render=True)
env = SparsePointEnv(goal_radius=0.5, max_episode_steps=100, goal_sampler='semi-circle', is_render=True)
for e in range(10):
    done = False
    obs = env.reset()
    total_rewards = 0
    step = 0
    while not done and step < env._max_episode_steps:
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        total_rewards += reward
        step += 1
        #if step % 10 == 0:
            # print(f"step: {step}, reward: {reward}")
        #env.render(mode='text', tick=None)
        #env.render(mode='rgb', tick=0.1)
        if done or step == env._max_episode_steps:
            break
            print(f"episode: {e}, total_rewards: {total_rewards}")
        
    
    