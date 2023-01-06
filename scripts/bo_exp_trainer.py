#!/usr/bin/env python
import gym
import safety_gym
import randomizer.safe_env
from safe_rl import cpo, cpo_within_BO, trpo_lagrangian_within_BO, ppo_lagrangian_within_BO, saute_ppo_lagrangian_whin_BO
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork, proc_id
import time


def saferl_trainer(seed, exp_name, cpu, env_name, env_kwargs=None, trainer_name='cpo',
                   exp_setup=None):
    # Hyperparameters
    if exp_setup is not None:
        num_steps = exp_setup['num_steps']
        steps_per_epoch = exp_setup['steps_per_epoch']
        epochs = int(num_steps / steps_per_epoch)
        save_freq = exp_setup['save_freq']
        target_kl = exp_setup['target_kl']
        cost_lim = exp_setup['cost_lim']
        max_ep_len = exp_setup['max_ep_len']

    else:
        num_steps = 6e5
        steps_per_epoch = 500
        epochs = int(num_steps / steps_per_epoch)
        save_freq = 50
        target_kl = 0.01
        cost_lim = 40
        max_ep_len = 200  # each episode length to determine whether the episode terminate caused by reaching the max episode length


    # Fork for parallelizing
    mpi_fork(cpu)
    # Prepare Logger
    logger_kwargs = setup_logger_kwargs(exp_name, seed)

    # Algo and Env
    env_name = env_name

    # define the trainer from cpo, ppo_Lagrangian and trpo_lagrangian
    if trainer_name == 'cpo':
        trainer = cpo_within_BO
    elif trainer_name == 'trpo':
        trainer = trpo_lagrangian_within_BO
    elif trainer_name == 'saute':
        trainer = saute_ppo_lagrangian_whin_BO
    elif trainer_name=='ppo':
        trainer = ppo_lagrangian_within_BO
    else:
        print('wrong trainer name!')

    trainer(env_fn=lambda: gym.make(env_name),
            ac_kwargs=dict(
                hidden_sizes=(256, 256),
            ),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            save_freq=save_freq,
            target_kl=target_kl,
            cost_lim=cost_lim,
            seed=seed,
            logger_kwargs=logger_kwargs,
            max_ep_len=max_ep_len,
            env_kwargs=env_kwargs
            )

    return logger_kwargs['output_dir']


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='test_cpo')
    parser.add_argument('--env_name', type=str, default='RandomizeSafeDoublePendulum-v0')
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--trainer', type=str, default='cpo')
    args = parser.parse_args()
    saferl_trainer(seed=args.seed, exp_name=args.exp_name, cpu=args.cpu, env_name=args.env_name,
                   trainer_name=args.trainer)
