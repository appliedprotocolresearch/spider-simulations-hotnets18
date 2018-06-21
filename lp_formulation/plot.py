import matplotlib.pyplot as plt
import numpy as np 

from utils import *

throughput = np.load('./throughput.npy')
total_flow_skew_list = np.load('./total_flow_skew.npy')

plt.plot(total_flow_skew_list, throughput)
plt.xlabel('Total allowed link imbalances')
plt.ylabel('Total throughput')

if SRC_TYPE is 'skew':
	plt.title('Skewed tx source (exp rate = ' + str(SKEW_RATE) + ')')
elif SRC_TYPE is 'Uniform':
	plt.title('Uniform tx source')
else:
	print "Error! Source type invalid."
plt.show()