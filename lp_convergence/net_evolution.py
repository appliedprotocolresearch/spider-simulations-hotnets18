""" program to simulate distributed flow and price update operations, 
and analyze their convergence behavior """

import copy
import networkx as nx 
import numpy as np 

from kshortestpaths import * 
from utils import *

class payment_network(object):
	def __init__(self, graph, demand_mat, credit_mat, delay, max_num_paths):
		""" initialize graph, demand matrix, link capacities """
		self.graph = copy.deepcopy(graph)
		self.demand_mat = demand_mat
		self.nonzero_demands = np.transpose(np.nonzero(self.demand_mat))
		self.credit_mat = credit_mat
		self.delay = delay
		self.capacity_mat = self.credit_mat * 1.0/self.delay
		assert np.shape(demand_mat)[0] == np.shape(demand_mat)[1]
		assert len(graph.nodes()) == np.shape(demand_mat)[0]

		""" initialize prices on links """
		self.link_prices_l = {}
		self.link_prices_y = {}
		self.link_prices_z = {}

		for i, j in self.nonzero_demands:
			self.link_prices_l[i, j] = 1.

		for e in self.graph.edges():
			self.link_prices_y[e[0], e[1]] = 1.
			self.link_prices_y[e[1], e[0]] = 1.
			self.link_prices_z[e[0], e[1]] = 1.
			self.link_prices_z[e[1], e[0]] = 1.			

		""" initalize flows on paths """
		self.total_srcdest_flow = {}
		self.link_flows = {}
		self.path_flows = {}
		self.paths = self.preselect_paths(max_num_paths)

		for i, j in self.nonzero_demands:
			self.total_srcdest_flow[i, j] = 0.
			self.path_flows[i, j] = {}
			for idx, path in enumerate(self.paths[i, j]):
				self.path_flows[i, j][idx] = 0.

		for e in self.graph.edges():
			self.link_flows[e[0], e[1]] = 0.
			self.link_flows[e[1], e[0]] = 0.

	def preselect_paths(self, max_num_paths):
		""" compute and store (at most) max_num_paths shortest paths for each source, 
		destination pair of nodes """ 
		paths = {}
		for i, j in self.nonzero_demands:
			paths[i, j] = ksp_yen(self.graph, i, j, max_num_paths)
		return paths

	def compute_path_price(self, path):
		""" return total price along path """
		i = path[0]
		j = path[-1]
		total_price = self.link_prices_l[i, j]
		for u, v in zip(path[:-1], path[1:]):
			total_price += self.link_prices_y[u, v]
			total_price += self.link_prices_z[u, v]
			total_price -= self.link_prices_z[v, u]
		return total_price

	def compute_path_slackness(self, path):
		""" return total slackness along path """
		i = path[0]
		j = path[-1]
		total_slack = (self.demand_mat[i, j] - self.total_srcdest_flow[i, j])*self.link_prices_l[i, j]
		for u, v in zip(path[:-1], path[1:]):
			total_slack += (self.capacity_mat[u, v] - self.link_flows[u, v])*self.link_prices_y[u, v]
			total_slack += (self.link_flows[v, u] - self.link_flows[u, v])*self.link_prices_z[u, v]
		return total_slack

	def update_flows(self):
		""" update flow variables depending on link prices """
		for i, j in self.nonzero_demands:
			for idx, path in enumerate(self.paths[i, j]):
				price = self.compute_path_price(path)
				temp_flow = self.path_flows[i, j][idx]
				self.path_flows[i, j][idx] += STEP_SIZE * (1. - price) # / (np.abs(1. - price) ** 2 + 0.01) 
				self.path_flows[i, j][idx] = np.max([0., self.path_flows[i, j][idx]])

				""" update total flow between i and j """
				self.total_srcdest_flow[i, j] -= temp_flow
				self.total_srcdest_flow[i, j] += self.path_flows[i, j][idx]

				""" update link flow states """
				for u, v in zip(path[:-1], path[1:]):
					self.link_flows[u, v] -= temp_flow
					self.link_flows[u, v] += self.path_flows[i, j][idx]

		# TODO: if max_num_paths is infinity, then update flow variables only on one shortest and longest path

	def update_prices(self):
		""" update price variables depending on link utilization """
		for i, j in self.nonzero_demands:
			if (self.demand_mat[i, j] - self.total_srcdest_flow[i, j]) > 0:
				self.link_prices_l[i, j] = 0.
			else:
				self.link_prices_l[i, j] = 4.

		for e in self.graph.edges():
			if (self.capacity_mat[e[0], e[1]] - self.link_flows[e[0], e[1]]) > 0:
				self.link_prices_y[e[0], e[1]] = 0.
			else:
				self.link_prices_y[e[0], e[1]] = 4.
			
			if (self.capacity_mat[e[1], e[0]] - self.link_flows[e[1], e[0]]) > 0:
				self.link_prices_y[e[1], e[0]] = 0.
			else:
				self.link_prices_y[e[1], e[0]] = 4.

			if (self.link_flows[e[1], e[0]] - self.link_flows[e[0], e[1]]) > 0:
				self.link_prices_z[e[0], e[1]] = 0.
			else:
				self.link_prices_z[e[0], e[1]] = 4.

			if (self.link_flows[e[0], e[1]] - self.link_flows[e[1], e[0]]) > 0:
				self.link_prices_z[e[1], e[0]] = 0.
			else:
				self.link_prices_z[e[1], e[0]] = 4.

	def print_flows(self):
		""" print current state of flows """
		for i, j in self.nonzero_demands:
			print "src: ", i, "dest: ", j
			for idx, path in enumerate(self.paths[i, j]):
				print "path: ", path, "flow: ", self.path_flows[i, j][idx]
			print " "

	def print_link_prices(self):
		""" print current state of prices """
		for i, j in self.nonzero_demands:
			print "l value for src: ", i, "dest: ", j, " is ", self.link_prices_l[i, j]
		print " "

		for e in self.graph.edges():
			print "y value for edge: ", e[0], e[1], " is ", self.link_prices_y[e[0], e[1]]
			print "y value for edge: ", e[1], e[0], " is ", self.link_prices_y[e[1], e[0]]
		print " "

		for e in self.graph.edges():
			print "z value for edge: ", e[0], e[1], " is ", self.link_prices_z[e[0], e[1]]
			print "z value for edge: ", e[1], e[0], " is ", self.link_prices_z[e[1], e[0]]
		print " "

	def print_path_prices(self):
		""" print total price along all paths """
		for i, j in self.nonzero_demands:
			print "src: ", i, "dest: ", j
			for idx, path in enumerate(self.paths[i, j]):
				total_price = self.link_prices_l[i, j]
				for u, v in zip(path[:-1], path[1:]):
					total_price += self.link_prices_y[u, v]
					total_price += self.link_prices_z[u, v]
					total_price -= self.link_prices_z[v, u]
				print "path: ", path, "price: ", total_price

	def print_primal_value(self):
		""" total flow sent in current state """
		total_flow = 0.
		for i, j in self.nonzero_demands:
			for idx, path in enumerate(self.paths[i, j]):
				total_flow += self.path_flows[i, j][idx]

		return total_flow

	def print_dual_value(self):
		""" total price in current state (same as dual objective) """
		total_price = 0.
		for i, j in self.nonzero_demands:
			total_price += self.link_prices_l[i, j]*self.demand_mat[i, j]

		for e in self.graph.edges():
			total_price += self.link_prices_y[e[0], e[1]]*self.capacity_mat[e[0], e[1]]
			total_price += self.link_prices_y[e[1], e[0]]*self.capacity_mat[e[1], e[0]]

		return total_price

	def print_errors(self):
		""" error in complementary slackness (CS) conditions """
		err_l = 0.
		err_y = 0.
		err_z = 0.
		err_x = 0.

		""" error in primal constraints (PC) """
		err_d = 0.
		err_c = 0.
		err_b = 0. 

		""" compute errors """						
		for i, j in self.nonzero_demands:
			err_d += np.max([0., self.total_srcdest_flow[i, j] - self.demand_mat[i, j]])
			err_l += np.abs(self.link_prices_l[i, j] * (self.demand_mat[i, j] - self.total_srcdest_flow[i, j]))

		for e in self.graph.edges():
			err_c += np.max([0., self.link_flows[e[0], e[1]] - self.capacity_mat[e[0], e[1]]]) 
			err_c += np.max([0., self.link_flows[e[1], e[0]] - self.capacity_mat[e[1], e[0]]]) 
			err_b += np.max([0., self.link_flows[e[1], e[0]] - self.link_flows[e[0], e[1]]])
			err_b += np.max([0., self.link_flows[e[0], e[1]] - self.link_flows[e[1], e[0]]])

			err_y += np.abs(self.link_prices_y[e[0], e[1]] * (self.capacity_mat[e[0], e[1]] - self.link_flows[e[0], e[1]]))
			err_y += np.abs(self.link_prices_y[e[1], e[0]] * (self.capacity_mat[e[1], e[0]] - self.link_flows[e[1], e[0]]))
			err_z += np.abs(self.link_prices_z[e[0], e[1]] * (self.link_flows[e[1], e[0]] - self.link_flows[e[0], e[1]]))
			err_z += np.abs(self.link_prices_z[e[1], e[0]] * (self.link_flows[e[0], e[1]] - self.link_flows[e[1], e[0]]))

		for i, j in self.nonzero_demands:
			for idx, path in enumerate(self.paths[i, j]):
				price = self.compute_path_price(path)
				err_x += np.abs(self.path_flows[i, j][idx] * (1. - price))

		return err_l, err_y, err_z, err_x, err_d, err_c, err_b

def main():

	""" type of graph """
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
	adj_mat = nx.adjacency_matrix(graph)

	""" type of demand matrix """
	if SRC_TYPE is 'uniform':
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
		print "Error! Source type invalid."			

	""" credits on links, delay and number of paths to consider """
	credit_mat = np.ones([n, n]) * 10
	credit_mat = adj_mat.multiply(credit_mat).todense()
	delay = .5
	max_num_paths = MAX_NUM_PATHS

	""" initialize payment network """
	network = payment_network(graph, demand_mat, credit_mat, delay, max_num_paths)

	""" initialize logs """
	primal_values = np.zeros([1, NUM_ITERATIONS])
	dual_values = np.zeros([1, NUM_ITERATIONS])
	cs_err_l = np.zeros([1, NUM_ITERATIONS])
	cs_err_y = np.zeros([1, NUM_ITERATIONS])
	cs_err_z = np.zeros([1, NUM_ITERATIONS])
	cs_err_x = np.zeros([1, NUM_ITERATIONS])
	pc_err_d = np.zeros([1, NUM_ITERATIONS])
	pc_err_c = np.zeros([1, NUM_ITERATIONS])
	pc_err_b = np.zeros([1, NUM_ITERATIONS])

	""" run distributed algorithm """
	for step in range(NUM_ITERATIONS):
		network.update_flows()
		primal_values[0, step] = network.print_primal_value()
		network.update_prices()
		dual_values[0, step] = network.print_dual_value()

		err_l, err_y, err_z, err_x, err_d, err_c, err_b = network.print_errors()
		cs_err_l[0, step] = err_l
		cs_err_y[0, step] = err_y
		cs_err_z[0, step] = err_z
		cs_err_x[0, step] = err_x
		pc_err_d[0, step] = err_d
		pc_err_c[0, step] = err_c 
		pc_err_b[0, step] = err_b 

	print primal_values
	print dual_values

	network.print_link_prices()
	network.print_flows()
	network.print_path_prices()

	""" save logs """
	np.save('./primal_values.npy', primal_values)	
	np.save('./dual_values.npy', dual_values)	
	np.save('./cs_err_l.npy', cs_err_l)
	np.save('./cs_err_y.npy', cs_err_y)
	np.save('./cs_err_z.npy', cs_err_z)
	np.save('./cs_err_x.npy', cs_err_x)
	np.save('./pc_err_d.npy', pc_err_d)
	np.save('./pc_err_c.npy', pc_err_c)
	np.save('./pc_err_b.npy', pc_err_b)

if __name__=='__main__':
	main()