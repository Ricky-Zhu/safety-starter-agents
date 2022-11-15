#!/usr/bin/env python

import time
import numpy as np
from safe_rl.utils.load_utils import load_policy
from safe_rl.utils.logx import EpochLogger
import gym
from scripts.bo_exp_trainer import cpo_trainer
from safe_rl.utils.mpi_tools import mpi_fork, proc_id
import randomizer.safe_env
from bayes_opt import BayesianOptimization
from scipy.optimize import NonlinearConstraint
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

count = 0  # record how many times the target function is invoked for naming the logger
default_cart = 0.1
default_pole = 0.6
training_record = []


def run_policy(env, get_action, max_ep_len=None, render=False):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :("
    num_episodes = 100
    logger = EpochLogger()
    o, r, d, ep_ret, ep_cost, ep_len, n = env.reset(), 0, False, 0, 0, 0, 0
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        a = np.clip(a, env.action_space.low, env.action_space.high)
        o, r, d, info = env.step(a)
        ep_ret += r
        ep_cost += info.get('cost', 0)
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=ep_len)
            # print('Episode %d \t EpRet %.3f \t EpCost %.3f \t EpLen %d' % (n, ep_ret, ep_cost, ep_len))
            o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpCost', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    # logger.dump_tabular()

    return logger.log_current_row.get("AverageEpRet", ""), logger.log_current_row.get("AverageEpCost", "")


def target_function(cart, pole):
    global count, default_cart, default_pole, training_record
    count += 1
    seed = 42
    exp_name = 'test_cpo_double_pendulum_{}'.format(count)
    cpu = 1
    env_name = 'RandomizeSafeDoublePendulum-v0'

    model_path = cpo_trainer(seed, exp_name, cpu, env_name, cart, pole)

    get_action, sess = load_policy(model_path,
                                   'last',
                                   deterministic=False)
    target_env = gym.make('RandomizeSafeDoublePendulum-v0')
    target_env.set_values(default_cart, default_pole)
    avg_return, avg_cost = run_policy(target_env, get_action, max_ep_len=200)
    print('evaluate on the default env. ret:{}, cost:{}'.format(avg_return, avg_cost))

    target_env.set_values(cart, pole)
    avg_return, avg_cost = run_policy(target_env, get_action, max_ep_len=200)
    print(avg_return, avg_cost)

    return avg_return, avg_cost


if __name__ == '__main__':
    avg_ret, avg_cost = target_function(0.12, 0.65)
    print('evaluate on the same env. ret:{}, cost:{}'.format(avg_ret,avg_cost))


