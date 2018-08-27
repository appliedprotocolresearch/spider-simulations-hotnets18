""" Convert graph topology to .dot file """
import networkx as nx 
import numpy as np

def get_dot_topology(graph, credit_mat):
	with open('./lightning.dot', 'w') as f:
		f.write('strict digraph lightning { \n')

		for v in graph.nodes():
			f.write('h' + str(v) + ' ' + '[type=host, mac="00:00:00:00:00:0' + str(v) + '", ip="10.0.0.' + str(v) + '"]; \n')

		for v in graph.nodes():
			f.write('s' + str(v) + ' ' + '[type=switch, id=' + str(v) + ', mac="20:00:00:00:00:0' + str(v) + '", ip="192.168.1.3"]; \n')

		for v in graph.nodes():
			f.write('s' + str(v) + ' -> ' + 'h' + str(v) + ' [capacity="10Gbps"]; \n')
			f.write('h' + str(v) + ' -> ' + 's' + str(v) + ' [capacity="10Gbps"]; \n')

		for u, v in graph.edges():
			f.write('s' + str(u) + ' -> ' + 's' + str(v) + ' [capacity="' + str(credit_mat[u, v]) + 
					'Gbps"]; \n')
			f.write('s' + str(v) + ' -> ' + 's' + str(u) + ' [capacity="' + str(credit_mat[v, u]) + 
					'Gbps"]; \n')			

		f.write('}')

def get_demand(graph, demand_mat):
	with open('./lightning.txt', 'w') as f:
		for i in graph.nodes():
			for j in graph.nodes():
				f.write(str(demand_mat[i, j]) + ' ')

def get_node_index(graph):
	with open('./lightning.hosts', 'w') as f:
		for v in graph.nodes():
			f.write('h' + str(v) + '\n')

def main():
	n = 30
	graph = nx.scale_free_graph(n, seed=11)
	graph = nx.Graph(graph)
	graph.remove_edges_from(graph.selfloop_edges())

	credit_mat = np.ones([n, n])
	demand_mat = np.ones([n, n]) * 1e6
	np.fill_diagonal(demand_mat, 0.0)

	get_dot_topology(graph, credit_mat)
	get_demand(graph, demand_mat)
	get_node_index(graph)

if __name__=='__main__':
	main()