import copy
import networkx as nx 
import numpy as np 
import parse
import cPickle as pickle 
import sys, os
import time

from gurobipy import *
from kshortestpaths import * 
from utils import *

class global_optimal_flows(object):
	def __init__(self, graph, demand_mat, credit_mat, max_num_paths):
		self.graph = copy.deepcopy(graph)
		self.demand_mat = demand_mat
		self.credit_mat = credit_mat
		self.max_path_length = 0
		self.delay = 0

		n = len(self.graph.nodes())
		assert np.shape(self.demand_mat) == (n, n)
		assert np.shape(self.credit_mat) == (n, n)
		print "demand matrix", np.sum(self.demand_mat)

		self.nonzero_demands = np.transpose(np.nonzero(self.demand_mat))

		self.m = Model()
		self.m.setParam('OutputFlag', 0)
		self.m.setParam('TimeLimit', TIME_LIMIT)

		time_var = time.time()

		self.pathflowVars = {}
		self.edgeskewVars = {}
		self.total_skew_constraint = None

		""" compute paths """
		self.paths = self.preselect_paths(max_num_paths)
		print "computed paths in time: ", time.time() - time_var

		""" compute delay: add 2 to include end-host links to routers """
		self.delay = 2*(self.max_path_length + 2) * SINGLE_HOP_DELAY

		""" create variables """
		for i, j in self.nonzero_demands:
			for idx, path in enumerate(self.paths[i, j]):
				self.pathflowVars[i, j, idx] = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, obj=1.0)

		for u, v in self.graph.edges():
			self.edgeskewVars[u, v] = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0.0)

		print "variables created in time: ", time.time() - time_var

		""" flow conservation constraints """
		for i, j in self.nonzero_demands:
			expr = 0.0
			for idx, path in enumerate(self.paths[i, j]):
				expr += self.pathflowVars[i, j, idx]
			self.m.addConstr(expr <= self.demand_mat[i, j])

		print "flow conservation constraints in time: ", time.time() - time_var

		""" capacity constraints due to credits and delay """
		for u, v in self.graph.edges():
			expr = 0.0
			flag = False
			for i, j in self.nonzero_demands:
				for idx, path in enumerate(self.paths[i, j]):
					trail = zip(path[:-1], path[1:])

					if LP_TYPE == 'pathdelta':
						trail_indices = [trail_idx for trail_idx, (p, q) in enumerate(trail) \
										if (p, q) == (u, v) or (p, q) == (v, u)]
						if trail_indices:
							delay_for_each_index = [2 + len(trail) + 1 + len(trail) - trail_idx for trail_idx in trail_indices]
							total_delay = sum(delay_for_each_index) * SINGLE_HOP_DELAY
							expr += self.pathflowVars[i, j, idx]*total_delay
							flag = True

					elif LP_TYPE == 'maxdelta':
						trail_indices = [trail_idx for trail_idx, (p, q) in enumerate(trail) \
										if (p, q) == (u, v) or (p, q) == (v, u)]
						if trail_indices:
							total_delay = len(trail_indices)
							expr += self.pathflowVars[i, j, idx]*total_delay
							flag = True

					else: 
						print "LP_TYPE invalid!"

			if LP_TYPE == 'pathdelta' and flag:
				self.m.addConstr(expr <= self.credit_mat[u, v])

			elif LP_TYPE == 'maxdelta' and flag:
				self.m.addConstr(expr <= self.credit_mat[u, v] * 1.0/self.delay)

			else:
				pass

		print "capacity constraints in time: ", time.time() - time_var

		""" flow skew constraints """
		for u, v in self.graph.edges():
			expr_right = 0.0
			expr_left = 0.0
			flag = False
			for i, j in self.nonzero_demands:
				for idx, path in enumerate(self.paths[i, j]):
					trail = zip(path[:-1], path[1:])

					trail_indices_left = [trail_idx for trail_idx, (p, q) in enumerate(trail) \
										  if (p, q) == (v, u)]
					trail_indices_right = [trail_idx for trail_idx, (p, q) in enumerate(trail) \
										  if (p, q) == (u, v)]

					if trail_indices_left:
						expr_left += self.pathflowVars[i, j, idx]*len(trail_indices_left)
						flag = True

					if trail_indices_right:
						expr_right += self.pathflowVars[i, j, idx]*len(trail_indices_right)
						flag = True

			if flag:
				self.m.addConstr(expr_right - expr_left <= self.edgeskewVars[u, v])
				self.m.addConstr(expr_left - expr_right <= self.edgeskewVars[u, v])

		print "skew constraints in time: ", time.time() - time_var

		""" update model """
		self.m.update()

	def preselect_paths(self, max_num_paths):
		""" compute and store (at most) max_num_paths shortest paths for each source, 
		destination pair of nodes """ 
		paths = {}
		for i, j in self.nonzero_demands:
			if PATH_TYPE is 'ksp':
				paths[i, j] = ksp_yen(self.graph, i, j, max_num_paths)
			elif PATH_TYPE is 'ksp_edge_disjoint':
				paths[i, j] = ksp_edge_disjoint(self.graph, i, j, max_num_paths)
			elif PATH_TYPE is 'kwp_edge_disjoint':
				paths[i, j] = kwp_edge_disjoint(self.graph, i, j, max_num_paths, self.credit_mat)
			elif PATH_TYPE is 'raeke':
				paths[i, j] = raeke(i, j)
			else:
				print "Error! Path type not found."

			""" keep track of path length to find max path length """
			for path in paths[i, j]:
				if len(path) > self.max_path_length:
					self.max_path_length = len(path)

		return paths

	def compute_lp_solution(self, total_flow_skew):
		""" update total skew constraint """
		if self.total_skew_constraint is not None:
			self.m.remove(self.total_skew_constraint)

		expr = 0.0
		for u, v in self.graph.edges():
			expr += self.edgeskewVars[u, v]
		self.total_skew_constraint = self.m.addConstr(expr <= total_flow_skew)

		self.m.update()
		self.m.setAttr("ModelSense", -1)
		self.m.optimize()

		obj = 0.0
		for i, j in self.nonzero_demands:
			for idx, path in enumerate(self.paths[i, j]):			
				obj += self.pathflowVars[i, j, idx].X

		return obj

	def print_lp_solution(self):
		""" print lp solution """
		for i, j in self.nonzero_demands:
			print "Demand ", i, j
			for idx, path in enumerate(self.paths[i, j]):
				print "path ", path, ":", self.pathflowVars[i, j, idx].X
			print " "

	def compute_capacity_slack(self):

		""" capacity constraints due to credits and delay """
		capacity_slack = {}
		capacity = {}
		for u, v in self.graph.edges():
			expr = 0.0
			flag = False
			for i, j in self.nonzero_demands:
				for idx, path in enumerate(self.paths[i, j]):
					trail = zip(path[:-1], path[1:])

					if LP_TYPE == 'pathdelta':
						trail_indices = [trail_idx for trail_idx, (p, q) in enumerate(trail) \
										if (p, q) == (u, v) or (p, q) == (v, u)]
						if trail_indices:
							delay_for_each_index = [2 + len(trail) + 1 + len(trail) - trail_idx for trail_idx in trail_indices]
							total_delay = sum(delay_for_each_index) * SINGLE_HOP_DELAY
							expr += self.pathflowVars[i, j, idx].X*total_delay
							flag = True

					elif LP_TYPE == 'maxdelta':
						trail_indices = [trail_idx for trail_idx, (p, q) in enumerate(trail) \
										if (p, q) == (u, v) or (p, q) == (v, u)]
						if trail_indices:
							total_delay = len(trail_indices)
							expr += self.pathflowVars[i, j, idx].X*total_delay
							flag = True

					else: 
						print "LP_TYPE invalid!"

			if LP_TYPE == 'pathdelta' and flag:
				capacity_slack[u, v] = self.credit_mat[u, v] - expr
				capacity[u, v] = self.credit_mat[u, v]

			elif LP_TYPE == 'maxdelta' and flag:
				capacity_slack[u, v] = self.credit_mat[u, v] * 1.0/self.delay - expr
				capacity[u, v] = self.credit_mat[u, v] * 1.0/self.delay

			else:
				pass

		return capacity_slack, capacity


	def draw_flow_graph(self):
		g = nx.Graph()
		g.add_nodes_from(range(len(self.graph.nodes())))		
		for i, j in self.nonzero_demands:
			for idx, path in enumerate(self.paths[i, j]):
				if self.pathflowVars[i, j, idx].X > 1e-6:
					for u, v in zip(path[:-1], path[1:]):
						g.add_edge(u, v)
		with open('./path_flow_graph.pkl', 'wb') as output:
			pickle.dump(g, output, pickle.HIGHEST_PROTOCOL)

	def print_paths_from_lp_solution(self, op_filename):
		""" output paths to file specific format for simulator parsing"""	
		filename = "optimal_paths/opt_" + op_filename
		with open(filename, 'w') as f:
			for i, j in self.nonzero_demands:
				f.write(str((i, j)) + " " + str(self.demand_mat[i, j]) +  "\n")
				for idx, path in enumerate(self.paths[i, j]):
					f.write(str(path) +  " " + str(self.pathflowVars[i, j, idx].X) + "\n")
				f.write("\n")
		f.close()		

def main():
				
	""" construct graph """
	if GRAPH_TYPE == 'test':
		graph = nx.Graph()
		graph.add_nodes_from([0, 1, 2, 3])
		graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
		n = len(graph.nodes())

	elif GRAPH_TYPE == 'scale_free':
		n = GRAPH_SIZE
		graph = nx.scale_free_graph(n, seed=RAND_SEED)
		graph = nx.Graph(graph)
		graph.remove_edges_from(graph.selfloop_edges())

	elif GRAPH_TYPE == 'small_world':
		n = GRAPH_SIZE
		graph = nx.watts_strogatz_graph(n, k=8, p=0.01, seed=RAND_SEED)

	elif GRAPH_TYPE == 'erdos_renyi':
		n = GRAPH_SIZE
		graph = nx.fast_gnp_random_graph(n, 0.2, seed=RAND_SEED)

	elif GRAPH_TYPE == 'isp':
		nodes, edges = parse.get_graph('../../speedy/data/visualizations/sample_topologies/BtNorthAmerica.gv')
		graph = nx.Graph()
		graph.add_nodes_from(nodes)
		graph.add_edges_from(edges)		
		n = len(graph.nodes())

	elif GRAPH_TYPE == 'lnd':
		graph = nx.read_edgelist("../oblivious_routing/lnd_dec4_2018_reducedsize.edgelist")
		rename_dict = {v: int(str(v)) for v in graph.nodes()}
		graph = nx.relabel_nodes(graph, rename_dict)
		for e in graph.edges():
			graph.edges[e]['capacity'] = int(str(graph.edges[e]['capacity']))
		graph = nx.Graph(graph)
		graph.remove_edges_from(graph.selfloop_edges())
		n = nx.number_of_nodes(graph)  		
		
	else:
		print "Error! Graph type invalid."

	assert nx.is_connected(graph)

	
	""" construct demand matrix """
	if SRC_TYPE == 'test':
		""" test load """
		demand_mat = np.zeros([n, n])
		demand_mat[0, 1] = 1.
		demand_mat[1, 0] = 1.
		demand_mat[1, 3] = 1.
		demand_mat[3, 1] = 1. 
		demand_mat = demand_mat / np.sum(demand_mat)
		demand_mat = demand_mat * 1000 * TXN_VALUE

	elif SRC_TYPE == 'uniform':
		""" uniform load """
		demand_mat = np.ones([n, n])
		np.fill_diagonal(demand_mat, 0.0)
		demand_mat = demand_mat / np.sum(demand_mat)
		demand_mat = demand_mat * 1000 * TXN_VALUE

	elif SRC_TYPE == 'skew':
		""" skewed load """
		exp_load = np.exp(np.arange(0, -n, -1) * SKEW_RATE)
		exp_load = exp_load.reshape([n, 1])
		demand_mat = exp_load * np.ones([1, n])
		np.fill_diagonal(demand_mat, 0.0)
		demand_mat = demand_mat / np.sum(demand_mat)
		demand_mat = demand_mat * 1000 * TXN_VALUE

	elif SRC_TYPE == 'lnd':
		""" lnd load """
		with open('./lnd_demand.pkl', 'rb') as input:
			demand_dict = pickle.load(input)
		demand_mat = np.zeros([n, n])
		for (i, j) in demand_dict.keys():
			demand_mat[i, j] = demand_dict[i, j]
		demand_mat = demand_mat * 10

	else:
		print "Error! Source type invalid."""


	""" construct credit matrix """
	if CREDIT_TYPE == 'uniform':
		credit_mat = np.ones([n, n]) * 10

	elif CREDIT_TYPE == 'random':
		np.random.seed(RAND_SEED)
		credit_mat = np.triu(np.random.rand(n, n), 1) * 2
		credit_mat += credit_mat.transpose()
		credit_mat = credit_mat.astype(int)

	elif CREDIT_TYPE == 'lnd':
		credit_mat = np.zeros([n, n])
		for e in graph.edges():
			credit_mat[e[0], e[1]] = graph.edges[e]['capacity']/1000
			credit_mat[e[1], e[0]] = graph.edges[e]['capacity']/1000

	else:
		print "Error! Credit matrix type invalid."

	total_flow_skew_list = [0.] # np.linspace(0, 2, 20)
	throughput = np.zeros(len(total_flow_skew_list))
	   
	solver = global_optimal_flows(graph, demand_mat, credit_mat, MAX_NUM_PATHS)

	for i, total_flow_skew in enumerate(total_flow_skew_list):
		throughput[i] = solver.compute_lp_solution(total_flow_skew)
		# solver.print_lp_solution()
		# capacity_slack, capacity = solver.compute_capacity_slack()
		# for key in capacity_slack.keys():
			# print key, capacity[key], round(capacity_slack[key], 2)

	# solver.draw_flow_graph()
	print throughput/np.sum(demand_mat)

	np.save('./throughput.npy', throughput)	
	np.save('./total_flow_skew.npy', total_flow_skew_list)

if __name__=='__main__':
	main()










