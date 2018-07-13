import matplotlib.pyplot as plt

# Num_shortest_paths vs total_flow

edge_based_opt = 0.24144351

max_num_paths = [1, 2, 4, 8, 16, 32, 64, 128, 256]
total_flow = [0.22454347, 0.24144351, 0.24144351, 0.24144351, 0.24144351, 0.24144351, 0.24144351, 0.24144351, 0.24144351]

plt.semilogx(max_num_paths, total_flow, label='path formulation')
plt.semilogx(max_num_paths, [edge_based_opt]*len(max_num_paths), '--', label='edge formulation')
plt.legend()
plt.xlabel('number of paths in lp')
plt.ylabel('total flow')
plt.show()
