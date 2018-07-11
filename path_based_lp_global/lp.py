import copy
import networkx as nx 
import numpy as np 
import parse
import pickle
import sys, os

from gurobipy import *
from kshortestpaths import * 
from utils import *

class global_optimal_flows(object):
	def __init__(self, graph, demand_mat, credit_mat, delay, max_num_paths):
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

		self.pathflowVars = {}
		self.edgeskewVars = {}
		self.total_skew_constraint = None
		self.paths = self.preselect_paths(max_num_paths)

		""" create variables """
		for i, j in self.nonzero_demands:
			for idx, path in enumerate(self.paths[i, j]):
				self.pathflowVars[i, j, idx] = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, obj=1.0)

		for u, v in self.graph.edges():
			self.edgeskewVars[u, v] = self.m.addVar(vtype=GRB.CONTINUOUS, lb=0.0)

		""" flow conservation constraints """
		for i, j in self.nonzero_demands:
			expr = 0.0
			for idx, path in enumerate(self.paths[i, j]):
				expr += self.pathflowVars[i, j, idx]
			self.m.addConstr(expr <= self.demand_mat[i, j])

		""" capacity constraints due to credits and delay """
		for u, v in self.graph.edges():
			expr = 0.0
			flag = False
			for i, j in self.nonzero_demands:
				for idx, path in enumerate(self.paths[i, j]):
					trail = zip(path[:-1], path[1:])
					if (u, v) in trail or (v, u) in trail:
						expr += self.pathflowVars[i, j, idx]
						flag = True
			if flag:
				self.m.addConstr(expr <= self.credit_mat[u, v] * 1.0/self.delay)

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
				self.m.addConstr(expr_right - expr_left <= self.edgeskewVars[u, v])
				self.m.addConstr(expr_left - expr_right <= self.edgeskewVars[u, v])

		""" update model """
		self.m.update()

	def preselect_paths(self, max_num_paths):
		""" compute and store (at most) max_num_paths shortest paths for each source, 
		destination pair of nodes """ 
		paths = {}
		for i, j in self.nonzero_demands:
			paths[i, j] = ksp_yen(self.graph, i, j, max_num_paths)
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

def read_demand_from_file(demand_file, num_nodes):
		demand_mat = np.zeros([num_nodes, num_nodes])
		count = 0
		with open(demand_file) as f:
			for line in f:
				parts = line.split(" ")
				src = int(parts[2])
				dst = int(parts[3])
				val = float(parts[1])
				demand_mat[src, dst] += val
				count += 1
		print demand_mat
		return demand_mat, count

def main():
		
	""" read credit amount from command line"""
	if len(sys.argv) == 3:
		credit_amt = int(sys.argv[2])
	else:
		credit_amt = 10

	""" construct output name based on demand file and credit"""
	demand_file = None
	op_filename = None
	if (len(sys.argv) >= 2):
		demand_file = sys.argv[1]
		base = os.path.basename(demand_file)
		op_filename = str(credit_amt) + os.path.splitext(base)[0]
		print op_filename

	if GRAPH_TYPE is 'scale_free':
		n = GRAPH_SIZE
		graph = nx.scale_free_graph(n)
		graph = nx.Graph(graph)

	elif GRAPH_TYPE is 'isp':
		nodes, edges = parse.get_graph('../../speedy/data/visualizations/sample_topologies/BtNorthAmerica.gv')
		graph = nx.Graph()
		graph.add_nodes_from(nodes)
		graph.add_edges_from(edges)
		n = len(graph.nodes())

	else:
		print "Error! Graph type invalid."

	if demand_file is not None:
		demand_mat, num_txns  = read_demand_from_file(demand_file, n)
		demand_mat = demand_mat/np.sum(demand_mat)
		#demand_mat = demand_mat/(float(num_txns)/1000)

	elif SRC_TYPE is 'uniform':
		""" uniform load """
		demand_mat = np.ones([n, n]) 
		np.fill_diagonal(demand_mat, 0.0)
		demand_mat = demand_mat/np.sum(demand_mat)		

	elif SRC_TYPE is 'skew':
		""" skewed load """
		exp_load = np.exp(np.arange(0, -n, -1) * SKEW_RATE)
		exp_load = exp_load.reshape([n, 1])
		demand_mat = exp_load * np.ones([1, n])
		np.fill_diagonal(demand_mat, 0.0)
		demand_mat = demand_mat/np.sum(demand_mat)
	else:
		print "Error! Source type invalid."""

	# graph = nx.Graph()
	# graph.add_nodes_from([0, 1, 2])
	# graph.add_edges_from([(0, 1), (1, 2), (0, 2)])
	# n = len(graph.nodes())
	# credit_amt = 1.
	# demand_mat = np.zeros([3, 3])
	# demand_mat[0, 1] = 1.
	# demand_mat[1, 2] = 1.
	# demand_mat[2, 0] = 1.
	# np.fill_diagonal(demand_mat, 0.0)
	# demand_mat = demand_mat/np.sum(demand_mat)			


	credit_mat = np.ones([n, n])*credit_amt
	delay = .5
	total_flow_skew_list = [0.] # np.linspace(0, 2, 20)
	throughput = np.zeros(len(total_flow_skew_list))
	   
	solver = global_optimal_flows(graph, demand_mat, credit_mat, delay, MAX_NUM_PATHS)

	for i, total_flow_skew in enumerate(total_flow_skew_list):
		throughput[i] = solver.compute_lp_solution(total_flow_skew)
		solver.print_lp_solution()

	print throughput

	np.save('./throughput.npy', throughput)	
	np.save('./total_flow_skew.npy', total_flow_skew_list)

if __name__=='__main__':
	main()










