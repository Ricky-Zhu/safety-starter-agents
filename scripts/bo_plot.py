import json
from bayes_opt.util import load_logs
import json
import matplotlib.pyplot as plt
import numpy as np


def plot(path):
    target = []
    constraint = []
    params = []

    # temporary
    cart_mean = []
    pole_mean = []
    cart_var = []
    pole_var = []
    with open(path, 'r') as j:
        while True:
            try:
                iteration = next(j)
            except StopIteration:
                break
            iteration = json.loads(iteration)
            target.append(iteration['target'])
            constraint.append(iteration['constraint'])
            params.append(iteration['params'])

            # temporary
            cart_mean.append(iteration['params']['cart_mean'])
            pole_mean.append(iteration['params']['pole_mean'])
            cart_var.append(iteration['params']['cart_var'])
            pole_var.append(iteration['params']['pole_var'])

    constraint = np.asarray(constraint)
    target = np.asarray(target)

    # temporary
    cart_mean = np.asarray(cart_mean)
    cart_var = np.asarray(cart_var)
    pole_mean = np.asarray(pole_mean)
    pole_var = np.asarray(pole_var)

    # fig, axs = plt.subplots(1, 1, figsize=(15, 10))

    # plot the constraint vs. target
    plt.scatter(constraint, target)
    plt.title('Constraint Vs Target')
    plt.xlabel('Constraint')
    plt.ylabel('Target')

    # plot the cart mean, pole mean and the target


path = '/home/ruiqi/bayrn_logs_2022-11-07_17-34-20.json'
plot(path)
plt.show()
