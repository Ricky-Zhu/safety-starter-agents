#!/usr/bin/env python
import os
import time
import imageio
import numpy as np
from safe_rl.utils.load_utils import load_policy
from safe_rl.utils.logx import EpochLogger
import gym
import safety_gym

def run_policy(env, get_action, max_ep_len=None, num_episodes=3, render=False, frame_save_path=None):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :("
    if frame_save_path is not None and not os.path.exists(frame_save_path):
        os.makedirs(frame_save_path)
    frame_count = 0

    logger = EpochLogger()
    o, r, d, ep_ret, ep_cost, ep_len, n = env.reset(), 0, False, 0, 0, 0, 0
    while n < num_episodes:
        if render:
            if frame_save_path is not None:
                frame = env.render(mode='rgb_array')
                frame_count += 1
                imageio.imwrite(frame_save_path + '/frame{}.jpeg'.format(str(frame_count).zfill(4)), frame)
            else:
                env.render()
            # time.sleep(1e-3)

        a = get_action(o)
        a = np.clip(a, env.action_space.low, env.action_space.high)
        o, r, d, info = env.step(a)
        ep_ret += r
        ep_cost += info.get('cost', 0)
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=ep_len)
            #print('Episode %d \t EpRet %.3f \t EpCost %.3f \t EpLen %d' % (n, ep_ret, ep_cost, ep_len))
            o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpCost', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=5)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    path = '/home/ruiqi/2023-01-01_19-15-19-ppo_lagrangian_DoggoGoal1_s0'
    get_action, sess = load_policy(path,
                                        args.itr if args.itr >= 0 else 'last',
                                        args.deterministic)
    env = gym.make('Safexp-DoggoGoal1-v0')
    env.seed(1)
    frame_path = path + '/frames'
    run_policy(env, get_action, args.len, args.episodes)
