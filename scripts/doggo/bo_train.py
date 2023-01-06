#!/usr/bin/env python

import time
import numpy as np
from safe_rl.utils.load_utils import load_policy
from safe_rl.utils.logx import EpochLogger
import gym, safety_gym
from scripts.bo_exp_trainer import saferl_trainer
from bayes_opt import BayesianOptimization
from scipy.optimize import NonlinearConstraint
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import time
from bayes_opt.util import load_logs

count = 0  # record how many times the target function is invoked for naming the logger
default_front_density = 0.5
default_rear_density = 0.5
training_record = []


def run_policy(env, get_action, max_ep_len=None, render=False):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :("
    num_episodes = 5
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


def target_function(front_density_mean, rear_density_mean, front_density_var, rear_density_var):
    global count, default_front_density, default_rear_density, training_record
    count += 1
    seed = 42
    cpu = 1  # for the BO target function only support cpu==1
    trainer_name = 'ppo'  # ppo->ppo lagrangian, trpo->trpo lagrangian, cpo->cpo
    exp_name = 'test_{}_doggo_{}'.format(trainer_name, count)

    env_name = 'RandomizeSafexp-DoggoGoal1-v0'

    # randomized parameters definition in the order means of all parameters and then variance of all parameters
    parameters = [front_density_mean, rear_density_mean, front_density_var, rear_density_var]
    env_kwargs = {'env_name': env_name,
                  'with_var': True,
                  'parameters': parameters}

    # to modify the hyper-parameters in the training, check out the cpo_trainer function
    exp_setup = {'num_steps': 1e8,
                 'steps_per_epoch': 60000,
                 'save_freq': 50,
                 'target_kl': 0.01,
                 'cost_lim': 25,
                 'max_ep_len': 1000}

    model_path = saferl_trainer(seed, exp_name, cpu, env_name, env_kwargs, trainer_name, exp_setup=exp_setup)

    get_action, sess = load_policy(model_path,
                                   'last',
                                   deterministic=False)
    target_env = gym.make(env_name)
    target_env.set_values(default_front_density, default_rear_density, 0.0, 0.0)
    avg_return, avg_cost = run_policy(target_env, get_action, max_ep_len=1000)
    print('finish one iteration')
    return avg_return, avg_cost


if __name__ == '__main__':
    def constraint_func(cart, pole):  # does not matter, only an input to the nonlinear constraint in scipy
        pass


    constraint_limit = 25.  # align with tht safety budget

    constraint = NonlinearConstraint(constraint_func, -np.inf, constraint_limit)

    # Bounded region of parameter space
    pbounds = {'front_density_mean': (0.2, 1.0), 'rear_density_mean': (0.2, 1.0), 'front_density_var': (9e-8, 2.25e-4),
               'rear_density_var': (1.66e-8, 1.66e-4)}

    optimizer = BayesianOptimization(
        f=target_function,
        constraint=constraint,
        pbounds=pbounds,
        verbose=0,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )

    # save logs
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    logger = JSONLogger(path="./bayrn_logs_doggo_{}.json".format(current_time))
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=5,
        n_iter=15,
    )
    print('save bo log bayrn_logs_doggo_{}.json!'.format(current_time))
