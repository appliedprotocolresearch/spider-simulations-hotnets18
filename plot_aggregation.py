import matplotlib.pyplot as plt
import numpy as np 



m = 1000.0
C = 100000.0
var_lambda = 1.0/m

alpha_range = range(1, int(m))

throughput = [((1 - m**2/((alpha**2)*(C**2)))**(np.log(m/alpha)))* var_lambda * m for alpha in alpha_range]

plt.plot(alpha_range[1:20], throughput[1:20])
plt.xlabel('Alpha')
plt.ylabel('Total throughput')

plt.show()
