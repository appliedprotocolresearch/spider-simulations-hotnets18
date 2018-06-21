import matplotlib.pyplot as plt
import numpy as np 

throughput = np.load('./throughput.npy')
total_flow_skew_list = np.load('./total_flow_skew.npy')

plt.plot(total_flow_skew_list, throughput)
plt.xlabel('Total allowed link imbalances')
plt.ylabel('Total throughput')
plt.title('Skewed tx source (exp rate = 0.5)')
# plt.title('Uniform tx source')
plt.show()