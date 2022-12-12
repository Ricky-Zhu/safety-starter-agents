#!/usr/bin/env python
import gym
import safety_gym
import randomizer.safe_env
from safe_rl import cpo, trpo_lagrangian, ppo_lagrangian, saute_trpo
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork, proc_id
import time
from randomizer.safe_env.wrappers.gcp import GCPWrapper


def saferl_trainer(seed, exp_name, cpu, env_name, env_kwargs=None, trainer_name='cpo', goal_condition=False,
                   cost_lim=40.,
                   **kwargs):
    # Hyperparameters
    num_steps = 6e5
    steps_per_epoch = 500
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 50
    target_kl = 0.01
    max_ep_len = 200  # each episode length to determine whether the episode terminate caused by reaching the max episode length

    # Fork for parallelizing
    mpi_fork(cpu)
    # Prepare Logger
    logger_kwargs = setup_logger_kwargs(exp_name, seed)

    # Algo and Env
    env_name = env_name

    # define the trainer from cpo, ppo_Lagrangian and trpo_lagrangian
    if trainer_name == 'cpo':
        trainer = cpo
    elif trainer_name == 'trpo':
        trainer = trpo_lagrangian
    elif trainer_name == 'saute':
        trainer = saute_trpo
        env_name = 'Saute' + env_name
    else:
        trainer = ppo_lagrangian

    if goal_condition:
        env_fn = lambda: GCPWrapper(gym.make(env_name, **kwargs))
    else:
        env_fn = lambda: gym.make(env_name, **kwargs)
    trainer(env_fn=env_fn,
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
    parser.add_argument('--exp_name', type=str, default='test_saute')
    parser.add_argument('--env_name', type=str, default='RandomizeSafeFetchSlideCostSimpleEnv-v0')
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--trainer', type=str, default='ppo')
    parser.add_argument('--cost_lim', type=float, default=40.)
    args = parser.parse_args()

    ######## the kwargs for goal conditional env ################
    fetch_slide_kwargs = dict(reward_type='dense',
                              )

    saferl_trainer(seed=args.seed, exp_name=args.exp_name, cpu=args.cpu, env_name=args.env_name,
                   trainer_name=args.trainer, cost_lim=args.cost_lim, goal_condition=True, **fetch_slide_kwargs)
