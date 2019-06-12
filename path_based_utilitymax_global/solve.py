import copy
import cvxpy as cvx 
import networkx as nx 
import numpy as np 
import parse
import cPickle as pickle 
import sys, os
import time

from kshortestpaths import * 
from utils import *

class global_optimal_flows(object):
	def __init__(self, graph, demand_mat, credit_mat, delay, max_num_paths, graph_type):
		self.graph = copy.deepcopy(graph)
		self.demand_mat = demand_mat
		self.credit_mat = credit_mat
		self.delay = delay

		n = len(self.graph.nodes())
		assert np.shape(self.demand_mat) == (n, n)
		assert np.shape(self.credit_mat) == (n, n)
		assert self.delay > 0.0
		print "demand matrix", np.sum(self.demand_mat)

		self.nonzero_demands = np.transpose(np.nonzero(self.demand_mat))

		# TODO: check if time limit can be set 
		self.objective = None 
		self.constraints = []
		self.problem = None 
		self.pathflowVars = {}
		self.edgeskewVars = {}
		self.total_skew_constraint = None

		time_var = time.time()

		""" compute paths """
		self.paths = self.preselect_paths(max_num_paths)
		with open('./k_shortest_paths.pkl', 'wb') as output:
			pickle.dump([self.paths, max_num_paths], output, pickle.HIGHEST_PROTOCOL)
		print "computed paths in time: ", time.time() - time_var

		""" create variables """
		for i, j in self.nonzero_demands:
			for idx, path in enumerate(self.paths[i, j]):
				self.pathflowVars[i, j, idx] = cvx.Variable()
				self.constraints.append(self.pathflowVars[i, j, idx] >= 0.)

		for u, v in self.graph.edges():
			self.edgeskewVars[u, v] = cvx.Variable()
			self.constraints.append(self.edgeskewVars[u, v] >= 0.)

		print "variables created in time: ", time.time() - time_var

		""" demand constraints """
		for i, j in self.nonzero_demands:
			expr = 0.0
			for idx, path in enumerate(self.paths[i, j]):
				expr += self.pathflowVars[i, j, idx]
			self.constraints.append(expr <= self.demand_mat[i, j])

		print "added flow conservation constraints in time: ", time.time() - time_var

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
				self.constraints.append(expr <= self.credit_mat[u, v])

			elif LP_TYPE == 'maxdelta' and flag:
				self.constraints.append(expr <= self.credit_mat[u, v] * 1.0/self.delay)

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
					if (u, v) in trail:
						expr_right += self.pathflowVars[i, j, idx]
						flag = True
					elif (v, u) in trail:
						expr_left += self.pathflowVars[i, j, idx]
						flag = True
					else:
						pass
			if flag:
				self.constraints.append(expr_right - expr_left <= self.edgeskewVars[u, v])
				self.constraints.append(expr_left - expr_right <= self.edgeskewVars[u, v])

		print "added skew constraints in time: ", time.time() - time_var

		""" objective """
		obj_expr = 0.0
		for i, j in self.nonzero_demands:
			expr = 0.0
			for idx, path in enumerate(self.paths[i, j]):
				expr += self.pathflowVars[i, j, idx]
			obj_expr += cvx.log(expr)
		self.objective = cvx.Maximize(obj_expr)

		""" problem """
		self.problem = cvx.Problem(self.objective, self.constraints)

		print "created problem in time: ", time.time() - time_var

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
				paths[i, j] = kwp_edge_disjoint(self.graph, i, j, max_num_paths, self.credit_mat, self.delay)
			else:
				print "Error! Path type not found."
		return paths

	def compute_lp_solution(self, total_flow_skew):
		""" update total skew constraint """
		if self.total_skew_constraint is not None:
			self.constraints.pop()

		expr = 0.0
		for u, v in self.graph.edges():
			expr += self.edgeskewVars[u, v]
		self.constraints.append(expr <= total_flow_skew)
		self.total_skew_constraint = True

		self.problem = cvx.Problem(self.objective, self.constraints)
		self.problem.solve()

		obj = 0.0
		for i, j in self.nonzero_demands:
			for idx, path in enumerate(self.paths[i, j]):
				obj += self.pathflowVars[i, j, idx].value 

		return obj

	def print_lp_solution(self):
		""" print lp solution """
		for i, j in self.nonzero_demands:
			print "Demand ", i, j
			for idx, path in enumerate(self.paths[i, j]):
				print "path ", path, ":", self.pathflowVars[i, j, idx].value 
			print " "

	def draw_flow_graph(self):
		g = nx.Graph()
		g.add_nodes_from(range(len(self.graph.nodes())))		
		for i, j in self.nonzero_demands:
			for idx, path in enumerate(self.paths[i, j]):
				if self.pathflowVars[i, j, idx].value > 1e-6:
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
					f.write(str(path) +  " " + str(self.pathflowVars[i, j, idx].value) + "\n")
				f.write("\n")
		f.close()		

def main():

	""" construct graph """
	if GRAPH_TYPE is 'test':
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
		graph = nx.watts_strogatz_graph(n, 8, 0.25, seed=RAND_SEED)
		graph = nx.Graph(graph) 		
		graph.remove_edges_from(graph.selfloop_edges())

	elif GRAPH_TYPE is 'isp':
		nodes, edges = parse.get_graph('../../speedy/data/visualizations/sample_topologies/BtNorthAmerica.gv')
		graph = nx.Graph()
		graph.add_nodes_from(nodes)
		graph.add_edges_from(edges)		
		n = len(graph.nodes())

	else:
		print "Error! Graph type invalid."


	""" construct demand matrix """
	if SRC_TYPE is 'test':
		""" test load """
		demand_mat = np.zeros([n, n])
		demand_mat[0, 1] = 1.
		demand_mat[1, 0] = 1.
		demand_mat[1, 3] = 1.
		demand_mat[3, 1] = 1.
		np.fill_diagonal(demand_mat, 0.0)
		demand_mat = demand_mat / np.sum(demand_mat)
		demand_mat = demand_mat * 1000 * TXN_VALUE

	elif SRC_TYPE is 'uniform':
		""" uniform load """
		demand_mat = np.ones([n, n])
		np.fill_diagonal(demand_mat, 0.0)
		demand_mat = demand_mat / np.sum(demand_mat)
		demand_mat = demand_mat * 1000 * TXN_VALUE

	elif SRC_TYPE is 'skew':
		""" skewed load """
		exp_load = np.exp(np.arange(0, -n, -1) * SKEW_RATE)
		exp_load = exp_load.reshape([n, 1])
		demand_mat = exp_load * np.ones([1, n])
		np.fill_diagonal(demand_mat, 0.0)
		demand_mat = demand_mat / np.sum(demand_mat)
		demand_mat = demand_mat * 1000 * TXN_VALUE

	elif SRC_TYPE == 'pickle':
		""" pickle load """
		with open('./demands/sw_10_routers_circ0_demand3_demand.pkl', 'rb') as input:
			demand_dict = pickle.load(input)
		demand_mat = np.zeros([n, n])
		for (i, j) in demand_dict.keys():
			demand_mat[i, j] = demand_dict[i, j]
		demand_mat = demand_mat * 88 * 3

	else:
		print "Error! Source type invalid."""


	""" construct credit matrix """
	if CREDIT_TYPE is 'uniform':
		credit_mat = np.ones([n, n]) * 1200

	else:
		print "Error! Credit matrix type invalid."


	delay = SINGLE_HOP_DELAY
	total_flow_skew_list = [0.] # np.linspace(0, 200000, 20)
	throughput = np.zeros(len(total_flow_skew_list))
	   
	solver = global_optimal_flows(graph, demand_mat, credit_mat, delay, MAX_NUM_PATHS, GRAPH_TYPE)

	for i, total_flow_skew in enumerate(total_flow_skew_list):
		throughput[i] = solver.compute_lp_solution(total_flow_skew)
		solver.print_lp_solution()

	# solver.draw_flow_graph()
	print throughput/np.sum(demand_mat)
	print solver.problem.status

	np.save('./throughput.npy', throughput)	
	np.save('./total_flow_skew.npy', total_flow_skew_list)

if __name__=='__main__':
	main()










