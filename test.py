from safe_rl import ppo_lagrangian
import gym, safety_gym
import randomizer.safe_env

env = gym.make('RandomizeSafeDoublePendulum-v0')

ppo_lagrangian(
	env_fn = lambda : gym.make('Safexp-PointGoal1-v0'),
	ac_kwargs = dict(hidden_sizes=(64,64))
	)