import matplotlib.pyplot as plt
import numpy as np 

from utils import *

primal_values = np.load('./primal_values.npy')
dual_values = np.load('./dual_values.npy')

iterations = np.zeros([1, np.shape(primal_values)[1]])
iterations[0, :] = range(np.shape(primal_values)[1])

plt.plot(iterations[0, :], primal_values[0, :])
plt.plot(iterations[0, :], dual_values[0, :])

# plt.show()
plt.savefig('plot.png')