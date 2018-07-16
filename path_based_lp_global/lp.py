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

		self.m = Model()
		self.m.setParam('OutputFlag', 0)
		self.m.setParam('TimeLimit', TIME_LIMIT)

		time_var = time.time()

		self.pathflowVars = {}
		self.edgeskewVars = {}
		self.total_skew_constraint = None

		""" compute paths """
		if graph_type == 'ripple' and USE_SAVED_PATHS:
			with open(SAVED_PATHS_PATH, 'rb') as input:
				[self.paths, _] = pickle.load(input)
		else:
			self.paths = self.preselect_paths(max_num_paths)
			with open('./k_shortest_paths.pkl', 'wb') as output:
				pickle.dump([self.paths, max_num_paths], output, pickle.HIGHEST_PROTOCOL)
			print "computed paths in time: ", time.time() - time_var

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

		print "flow conservation constraints: ", time.time() - time_var

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

		print "capacity constraints: ", time.time() - time_var

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

		print "skew constraints: ", time.time() - time_var

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
				paths[i, j] = kwp_edge_disjoint(self.graph, i, j, max_num_paths, self.credit_mat, self.delay)
			else:
				print "Error! Path type not found."
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
		
	""" read credit amount from command line"""
	if len(sys.argv) == 3:
		credit_amt = int(sys.argv[2])
	else:
		credit_amt = CREDIT_AMT

	""" construct output name based on demand file and credit"""
	demand_file = None
	op_filename = None
	if (len(sys.argv) >= 2):
		demand_file = sys.argv[1]
		base = os.path.basename(demand_file)
                op_filename = str(credit_amt) + os.path.splitext(base)[0]
		print op_filename

	""" construct graph """
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

	elif GRAPH_TYPE is 'ripple':
		adjacent, credits = parse.parse_credit_link_graph(RIPPLE_CREDIT_PATH)
		nodes, edges = parse.convert_adj_dict_to_list(adjacent)
		graph = nx.Graph()
		graph.add_nodes_from(nodes)
		graph.add_edges_from(edges)
                credit_amt = 0
                op_filename = "0RippleStaticClean_45000_Tr" if op_filename is None else op_filename	
                n = len(graph.nodes())
		
	else:
		print "Error! Graph type invalid."

	""" construct demand matrix """
	if demand_file is not None:
		demand_mat, num_txns  = parse.read_demand_from_file(demand_file, n)
		demand_mat = demand_mat/np.sum(demand_mat)

		if 'demandMatrix' not in demand_file:
			demand_mat = demand_mat/(float(num_txns)/1000)

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

	elif SRC_TYPE is 'ripple':
		assert GRAPH_TYPE is 'ripple'
		demand_mat, num_txns = parse.read_demand_from_file(RIPPLE_TXN_PATH, n)
		demand_mat = demand_mat/(float(num_txns)/1000.)

	else:
		print "Error! Source type invalid."""

	""" construct credit matrix """
	if CREDIT_TYPE is 'uniform':
		credit_mat = np.ones([n, n])*credit_amt

	elif CREDIT_TYPE is 'ripple':
		assert GRAPH_TYPE is 'ripple'
		assert SRC_TYPE is 'ripple'
		credit_mat = parse.convert_credit_dict_to_mat(credits, n)

	else:
		print "Error! Credit matrix type invalid."

	delay = .5
	total_flow_skew_list = [0.] # np.linspace(0, 2, 20)
	throughput = np.zeros(len(total_flow_skew_list))
	   
	solver = global_optimal_flows(graph, demand_mat, credit_mat, delay, MAX_NUM_PATHS, GRAPH_TYPE)

	for i, total_flow_skew in enumerate(total_flow_skew_list):
		throughput[i] = solver.compute_lp_solution(total_flow_skew)
		#solver.print_lp_solution()

	# solver.draw_flow_graph()
	print throughput/np.sum(demand_mat)

	if op_filename is not None:
		solver.print_paths_from_lp_solution(op_filename)
		obj_output_filename = "/home/ubuntu/lightning_routing/speedy/src/optimal_paths/"
		obj_output_filename += "obj_" + op_filename
		f = open(obj_output_filename, "a")
		f.write(str(throughput[0]) + " " + str(np.sum(demand_mat)) + " ")
		f.close()

	np.save('./throughput.npy', throughput)	
	np.save('./total_flow_skew.npy', total_flow_skew_list)

if __name__=='__main__':
	main()










