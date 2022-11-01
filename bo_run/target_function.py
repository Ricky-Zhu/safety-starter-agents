#!/usr/bin/env python
import gym
import safety_gym
import randomizer.safe_env
from safe_rl import cpo
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork


def target_function(cart, pole):
    # Hyperparameters
    num_steps = 1e6
    steps_per_epoch = 4000
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 50
    target_kl = 0.01
    cost_lim = 40

    # Fork for parallelizing
    cpu = 4
    mpi_fork(cpu)

    # Prepare Logger
    exp_name = 'test'
    seed = 42
    logger_kwargs = setup_logger_kwargs(exp_name, seed)

    # Algo and Env
    env_name = 'RandomizeSafeDoublePendulum-v0'

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


