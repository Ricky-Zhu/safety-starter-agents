import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot(data_dir, smooth=10, title=None):
    data = pd.read_csv(data_dir, sep='\t')
    avg_returns = data['AverageEpRet'].to_numpy()
    avg_costs = data['AverageEpCost'].to_numpy()
    interaction_nums = data['TotalEnvInteracts'].to_numpy()

    # get the legal max avg_ret
    valid_ind = np.where(avg_returns>50.)[0]
    valid_return = avg_returns[valid_ind]
    ind = np.where(valid_return>97.4)[0]
    print(interaction_nums[ind[0]])

    # smooth the data
    def smooth_data(data, smooth):
        if smooth > 1:
            y = np.ones(smooth)
            x = np.asarray(data)
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            return smoothed_x
        else:
            return data

    avg_returns = smooth_data(avg_returns, smooth)
    avg_costs = smooth_data(avg_costs, smooth)
    plt.plot(interaction_nums, avg_returns, label='avg_returns')
    plt.plot(interaction_nums, avg_costs, label='avg_costs')
    plt.ticklabel_format(style='sci', scilimits=(0, 2), axis='x')
    plt.xlabel('interaction')
    if title is not None:
        plt.title(title)
    plt.legend()


path = '/home/ruiqi/temp/2022-11-29_test_3_baselines/2022-11-29_17-14-16-test_3_baselines_s0/progress.txt'
plot(path, smooth=-1, title='Lagrangian PPO')
plt.show()
