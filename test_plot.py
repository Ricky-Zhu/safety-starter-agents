import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot(path):
    data = pd.read_csv(path,sep='\t')
    avg_return = data['AverageEpRet'].to_numpy()
    avg_cost = data['AverageEpCost'].to_numpy()
    length = data['TotalEnvInteracts'].to_numpy()
    plt.plot(length,avg_return)
    plt.plot(length,avg_cost)

path = '/home/ruiqi/2022-11-01_11-12-28-test_cpo_s42/progress.txt'
plot(path)
plt.show()
