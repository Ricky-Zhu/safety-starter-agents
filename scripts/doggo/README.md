# Safe RL Transfer with Constraint BO in env Doggo

Usage:
1. define the search space for the two domain parameters: front body density, rear body density
```python
# modify the search space
pbounds = {'front_density_mean': (0.2, 1.0), 'rear_density_mean': (0.2, 1.0), 'front_density_var': (9e-8, 2.25e-4),
               'rear_density_var': (1.66e-8, 1.66e-4)}
```

2. if want to modify the experiment parameters, check following in `bo_train.py`

```python
exp_setup = {'num_steps': 1e8,
                 'steps_per_epoch': 60000,
                 'save_freq': 50,
                 'target_kl': 0.01,
                 'cost_lim': 25,
                 'max_ep_len': 1000}
```

3. to modify the learning parameters of the agent (e.g. value network learning rate), check the specific class such as `PPOAgent()` or the 
function in the `run_agent.py` (e.g. run_polopt_agent_within_BO)
run the script


4. run the experiment by cmd: `python bo_train.py`