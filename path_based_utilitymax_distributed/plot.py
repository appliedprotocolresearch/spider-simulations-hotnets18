import matplotlib.pyplot as plt
import numpy as np 

from utils import *

subsample = 20

"""read logs """
total_flow_values = np.load('./total_flow_values.npy')

cs_err_l = np.load('./cs_err_l.npy')
cs_err_y = np.load('./cs_err_y.npy')
cs_err_z = np.load('./cs_err_z.npy')

pc_err_d = np.load('./pc_err_d.npy')
pc_err_c = np.load('./pc_err_c.npy')
pc_err_b = np.load('./pc_err_b.npy')

iterations = np.zeros([1, np.shape(primal_values)[1]])
iterations[0, :] = range(np.shape(primal_values)[1])

""" plot """
plt.plot(iterations[0, ::subsample], primal_values[0, ::subsample], '.', label='total flow value')

# plt.plot(iterations[0, ::subsample], cs_err_l[0, ::subsample], '-s', label='cs_err_l')
# plt.plot(iterations[0, ::subsample], cs_err_y[0, ::subsample], '-^', label='cs_err_y')
# plt.plot(iterations[0, ::subsample], cs_err_z[0, ::subsample], 'D', label='cs_err_z')

# plt.plot(iterations[0, ::subsample], pc_err_d[0, ::subsample], '-+', label='pc_err_d')
# plt.plot(iterations[0, ::subsample], pc_err_c[0, ::subsample], '-+', label='pc_err_c')
# plt.plot(iterations[0, ::subsample], pc_err_b[0, ::subsample], '-+', label='pc_err_b')

plt.legend()
plt.show()
# plt.savefig('plot.png')