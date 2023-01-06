# Safe RL Transfer with Constraint BO in env Doggo

Usage:

```python
# modify the search space
pbounds = {'front_density_mean': (0.2, 1.0), 'rear_density_mean': (0.2, 1.0), 'front_density_var': (9e-8, 2.25e-4),
               'rear_density_var': (1.66e-8, 1.66e-4)}
```

run the script

`python bo_train.py`