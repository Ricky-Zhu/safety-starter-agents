import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

color = sns.color_palette()
sns.set_style('darkgrid')

# the target domain data required for reaching the best performance found by the safeBO
# [cpo,lagrangian_PPO,lagrangian_TRPO]
# data = iteration_num x evaluation_num * episode_length = 20 x 5 x 200 = 20000
data_quant = np.array([20000]).repeat(3)
data_req_ori = np.array([1.2e5, 1.5e5, 2.1e5])

ratio_data = data_req_ori / data_quant

############################
# the best performance found by the safeBO
performances_safe_BO = np.array([97.40, 67.13, 70.74])
# the performance of directly train on the env
optimal_performances = np.array([98.96516, 93.26, 100.72])

performance_ratio = performances_safe_BO / optimal_performances

# plot
index = ['CPO', 'Lagrangian PPO', 'Lagrangian TRPO']
# data ratio plot
data_plot = sns.barplot(index, performance_ratio, alpha=0.8, color=color[2])
for i in data_plot.containers:
    data_plot.bar_label(i, )

plt.show()
