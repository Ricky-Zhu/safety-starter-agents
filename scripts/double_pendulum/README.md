# Safe RL Transfer with Constraint BO in env randomize double pendulum

Usage:
1. define the search space for the two domain parameters: cart size, pole size
```python
# modify the search space
pbounds = {'cart_mean': (0.05, 0.15), 'pole_mean': (0.55, 0.85), 'cart_var': (9e-8, 2.25e-4),
               'pole_var': (1.66e-8, 1.66e-4)}
```

2. if want to modify the experiment parameters, check following in `bo_train.py`

```python
exp_setup = {'num_steps': 6e5,
                 'steps_per_epoch': 1000,
                 'save_freq': 50,
                 'target_kl': 0.01,
                 'cost_lim': 40,
                 'max_ep_len': 200}
```

3. to modify the learning parameters of the agent (e.g. value network learning rate), check the specific class such as `PPOAgent()` or the 
function in the `run_agent.py` (e.g. run_polopt_agent_within_BO)
run the script


4. run the experiment by cmd: `python bo_train.py`