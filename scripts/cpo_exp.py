#!/usr/bin/env python
import gym
import safety_gym
import randomizer.safe_env
from safe_rl import cpo
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork


def main(seed, exp_name, cpu, env_name):
    # Hyperparameters
    num_steps = 1e6
    steps_per_epoch = 4000
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 50
    target_kl = 0.01
    cost_lim = 40

    # Fork for parallelizing
    mpi_fork(cpu)

    # Prepare Logger
    logger_kwargs = setup_logger_kwargs(exp_name, seed)

    # Algo and Env
    env_name = env_name

    cpo(env_fn=lambda: gym.make(env_name),
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
        max_ep_len=200
        )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='test_cpo')
    parser.add_argument('--env_name', type=str, default='RandomizeSafeDoublePendulum-v0')
    parser.add_argument('--cpu', type=int, default=1)
    args = parser.parse_args()

    main(args.seed, args.exp_name, args.cpu, args.env_name)
