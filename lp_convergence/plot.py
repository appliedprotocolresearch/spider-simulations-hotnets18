import matplotlib.pyplot as plt
import numpy as np 

from utils import *

primal_values = np.load('./primal_values.npy')
dual_values = np.load('./dual_values.npy')

cs_err_l = np.load('./cs_err_l.npy')
cs_err_y = np.load('./cs_err_y.npy')
cs_err_z = np.load('./cs_err_z.npy')
cs_err_x = np.load('./cs_err_x.npy')

iterations = np.zeros([1, np.shape(primal_values)[1]])
iterations[0, :] = range(np.shape(primal_values)[1])

plt.plot(iterations[0, :], primal_values[0, :], label='primal value')
# plt.plot(iterations[0, :], dual_values[0, :], label='dual value')

plt.plot(iterations[0, :], cs_err_l[0, :], label='cs_err_l')
plt.plot(iterations[0, :], cs_err_y[0, :], label='cs_err_y')
plt.plot(iterations[0, :], cs_err_z[0, :], label='cs_err_z')
plt.plot(iterations[0, :], cs_err_x[0, :], label='cs_err_x')

plt.legend()
# plt.show()
plt.savefig('plot.png')