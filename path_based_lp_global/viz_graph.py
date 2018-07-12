import networkx as nx
import matplotlib.pyplot as plt
import cPickle as pickle 

with open('./path_flow_graph.pkl', 'rb') as input:
	g = pickle.load(input)

# with open('./isp_graph.pkl', 'rb') as input:
# 	g = pickle.load(input)

# pos = dict()
# pos.update((i, (1, i)) for i in range(15))
# pos.update((i+15, (2, i)) for i in range(15))
# pos = nx.spring_layout(g, scale=2)
# pos = nx.random_layout(g)
# pos = nx.shell_layout(g)
pos = nx.circular_layout(g)
# pos = nx.spectral_layout(g)
nx.draw(g, pos, with_labels=True, node_color=g.nodes())
plt.show()