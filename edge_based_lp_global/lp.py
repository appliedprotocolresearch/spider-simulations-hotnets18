import copy
import networkx as nx 
import numpy as np 
import parse
import pickle
import sys, os
import time

from gurobipy import *
from utils import *

class global_optimal_flows(object):
	def __init__(self, graph, demand_mat, credit_mat, delay):
		self.graph = copy.deepcopy(graph)
		self.demand_mat = demand_mat
		self.credit_mat = credit_mat
		self.delay = delay

		n = len(self.graph.nodes())
		assert np.shape(self.demand_mat) == (n, n)
		assert np.shape(self.credit_mat) == (n, n)
		assert self.delay > 0.0

		self.nonzero_demands = np.transpose(np.nonzero(self.demand_mat))

		self.m = Model()
		self.m.setParam('OutputFlag', 0)
		self.m.setParam('TimeLimit', TIME_LIMIT)

		time_var = time.time()

		self.edgeflowVars = {}
		self.nodeflowVars = {}
		self.edgeskewVars = {}
		self.total_skew_constraint = None

		""" create variables """
		for i, j in self.nonzero_demands:
			if i != j:
				self.nodeflowVars[i, j] = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=self.demand_mat[i, j], obj=1.0)
			else:
				self.nodeflowVars[i, j] = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=self.demand_mat[i, j])
			
			for u, v in self.graph.edges():
				self.edgeflowVars[i, j, u, v] = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
				self.edgeflowVars[i, j, v, u] = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0.0)

		for u, v in self.graph.edges():
			self.edgeskewVars[u, v] = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0.0)

		print "variables created in time: ", time.time() - time_var

		""" flow conservation constraints """
		for i, j in self.nonzero_demands:
			expr_in = 0.0
			expr_out = 0.0
			for u in self.graph.neighbors(i):
				expr_in += self.edgeflowVars[i, j, u, i]
				expr_out += self.edgeflowVars[i, j, i, u]
			self.m.addConstr(expr_out - expr_in == self.nodeflowVars[i, j])

			expr_in = 0.0
			expr_out = 0.0
			for u in self.graph.neighbors(j):
				expr_in += self.edgeflowVars[i, j, u, j]
				expr_out += self.edgeflowVars[i, j, j, u]
			self.m.addConstr(expr_in - expr_out == self.nodeflowVars[i, j])

			for v in self.graph.nodes():
				if v != i and v!= j:
					expr_in = 0.0
					expr_out = 0.0
					for u in self.graph.neighbors(v):
						expr_in += self.edgeflowVars[i, j, u, v]
						expr_out += self.edgeflowVars[i, j, v, u]
					self.m.addConstr(expr_in == expr_out)

		print "Flow conservation constraints in time: ", time.time() - time_var

		""" flow constraints due to credits and delay """
		for u, v in self.graph.edges():
			expr = 0.0
			for i, j in self.nonzero_demands:
				expr += self.edgeflowVars[i, j, u, v]
				expr += self.edgeflowVars[i, j, v, u]
			self.m.addConstr(expr <= self.credit_mat[u, v] * 1.0/self.delay)

		print "Flow constraints due to credits and delay in time: ", time.time() - time_var

		""" flow skew constraints """
		for u, v in self.graph.edges():
			expr_right = 0.0
			expr_left = 0.0
			for i, j in self.nonzero_demands:
				expr_right += self.edgeflowVars[i, j, u, v]
				expr_left += self.edgeflowVars[i, j, v, u]
			self.m.addConstr(expr_right - expr_left <= self.edgeskewVars[u, v])
			self.m.addConstr(expr_left - expr_right <= self.edgeskewVars[u, v])

		print "Flow skew constraints in time: ", time.time() - time_var

		""" update model """
		self.m.update()

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
			obj += self.nodeflowVars[i, j].X

		return obj

	def print_lp_solution(self):
		""" print lp solution """
		for i, j in self.nonzero_demands:
			print "Demand ", i, j
			for u, v in self.graph.edges():
				print "edge ", u, v, ":", self.edgeflowVars[i, j, u, v].X, ":", self.edgeflowVars[i, j, v, u].X
		print " "

	def compute_paths_from_lp_solution(self):
		""" compute and return paths """
		graph = self.graph.to_directed()
		lp_paths = {}
		for i, j in self.nonzero_demands:
			commodity_graph = copy.deepcopy(graph)
			edge_dict = {}
			for u, v in graph.edges():
				if self.edgeflowVars[i, j, u, v].X > 0:
					print("some flow is non-zero")
					edge_dict[u, v] = self.edgeflowVars[i, j, u, v].X
				else:
					commodity_graph.remove_edge(u, v)
			nx.set_edge_attributes(commodity_graph, 'lp_weights', edge_dict)

			lp_paths[i, j] = []
			while nx.has_path(commodity_graph, i, j):
				path = peel_path(commodity_graph, i, j)
				lp_paths[i, j].append(path)

		return lp_paths

	def print_paths_from_lp_solution(self, op_filename):
		""" compute and output paths in specific format"""
		lp_paths = self.compute_paths_from_lp_solution()
		filename = "optimal_paths/opt_" + op_filename 
		with open(filename, 'w') as f:
			for key, value in lp_paths.items():
				f.write(str(key) + " " + str(self.demand_mat[key]) +  "\n")
				f.write("Value is " + str(value))
				for v in value:
					f.write(str(v['path']) +  " " + str(v['weight']) + "\n")
				f.write("\n")
		f.close()

def peel_path(commodity_graph, i, j):
	path = nx.dijkstra_path(commodity_graph, i, j)
	print ("peeling path")

	weights = [commodity_graph[u][v]['lp_weights'] for u, v in zip(path[:-1], path[1:])]
	min_weight = np.min(weights)

	for u, v in zip(path[:-1], path[1:]):
		if commodity_graph[u][v]['lp_weights'] == min_weight:
			commodity_graph.remove_edge(u, v)
		else:
			commodity_graph[u][v]['lp_weights'] -= min_weight

	path_dict = {}
	path_dict['path'] = path
	path_dict['weight'] = min_weight
	return path_dict


def main():

	""" read credit amount """
	credit_amt = CREDIT_AMT


	""" construct graph """
	if GRAPH_TYPE is 'test':
		graph = nx.Graph()
		graph.add_nodes_from([0, 1, 2, 3])
		graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
		n = len(graph.nodes())

	elif GRAPH_TYPE is 'scale_free':
		n = GRAPH_SIZE
		graph = nx.scale_free_graph(n, seed=RAND_SEED)
		graph = nx.Graph(graph)
		graph.remove_edges_from(graph.selfloop_edges())

	elif GRAPH_TYPE is 'erdos_renyi':
		n = GRAPH_SIZE
		graph = nx.fast_gnp_random_graph(n, 0.2, seed=RAND_SEED)

	elif GRAPH_TYPE is 'isp':
		nodes, edges = parse.get_graph('../../speedy/data/visualizations/sample_topologies/BtNorthAmerica.gv')
		graph = nx.Graph()
		graph.add_nodes_from(nodes)
		graph.add_edges_from(edges)		
		n = len(graph.nodes())
		
	else:
		print "Error! Graph type invalid."

	assert nx.is_connected(graph)


	""" construct demand matrix """
	if SRC_TYPE is 'test':
		""" test load """
		demand_mat = np.zeros([n, n])
		demand_mat[0, 1] = 1.
		demand_mat[1, 0] = 1.
		demand_mat[1, 3] = 1.
		demand_mat[3, 1] = 1. 
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

	else:
		print "Error! Source type invalid."""


	""" construct credit matrix """
	if CREDIT_TYPE is 'uniform':
		credit_mat = np.ones([n, n])*credit_amt

	elif CREDIT_TYPE is 'random':
		np.random.seed(RAND_SEED)
		credit_mat = np.triu(np.random.rand(n, n), 1) * 2 * credit_amt
		credit_mat += credit_mat.transpose()

	else:
		print "Error! Credit matrix type invalid."


	delay = DELAY
	total_flow_skew_list = [0.] # np.linspace(0, 2, 20)
	throughput = np.zeros(len(total_flow_skew_list))
	   
	solver = global_optimal_flows(graph, demand_mat, credit_mat, delay)

	for i, total_flow_skew in enumerate(total_flow_skew_list):
		throughput[i] = solver.compute_lp_solution(total_flow_skew)
		# solver.print_lp_solution()

	print throughput/np.sum(demand_mat)

	np.save('./throughput.npy', throughput)	
	np.save('./total_flow_skew.npy', total_flow_skew_list)

if __name__=='__main__':
	main()










