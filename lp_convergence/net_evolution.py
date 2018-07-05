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

	def update_flows(self):
		""" update flow variables depending on link prices """
		for i, j in self.nonzero_demands:
			for idx, path in enumerate(self.paths[i, j]):
				price = self.compute_path_price(path)
				temp_flow = self.path_flows[i, j][idx]
				self.path_flows[i, j][idx] += STEP_SIZE * (1. - price) # TODO: change precise update rule later
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
			self.link_prices_l[i, j] -= STEP_SIZE * (self.demand_mat[i, j] - self.total_srcdest_flow[i, j])
			self.link_prices_l[i, j] = np.max([0., self.link_prices_l[i, j]])

		for e in self.graph.edges():
			self.link_prices_y[e[0], e[1]] -= STEP_SIZE * (self.capacity_mat[e[0], e[1]] - self.link_flows[e[0], e[1]])
			self.link_prices_y[e[0], e[1]] = np.max([0., self.link_prices_y[e[0], e[1]]])
			self.link_prices_y[e[1], e[0]] -= STEP_SIZE * (self.capacity_mat[e[1], e[0]] - self.link_flows[e[0], e[1]])
			self.link_prices_y[e[1], e[0]] = np.max([0., self.link_prices_y[e[1], e[0]]])
			self.link_prices_z[e[0], e[1]] -= STEP_SIZE * (self.link_flows[e[1], e[0]] - self.link_flows[e[0], e[1]])
			self.link_prices_z[e[0], e[1]] = np.max([0., self.link_prices_z[e[0], e[1]]])
			self.link_prices_z[e[1], e[0]] -= STEP_SIZE * (self.link_flows[e[0], e[1]] - self.link_flows[e[1], e[0]])
			self.link_prices_z[e[1], e[0]] = np.max([0., self.link_prices_z[e[1], e[0]]])

	def print_flows(self):
		""" print current state of flows """
		for i, j in self.nonzero_demands:
			print "src: ", i, "dest: ", j
			for idx, path in enumerate(self.paths[i, j]):
				print "path: ", path, "flow: ", self.path_flows[i, j][idx]
			print " "

	def print_prices(self):
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

def main():
	# load ISP graph 
	# initialize payment_network

	graph = nx.Graph()
	graph.add_nodes_from([0, 1, 2, 3, 4])
	graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])

	demand_mat = np.ones([5, 5]) / 25.
	np.fill_diagonal(demand_mat, 0.0)

	credit_mat = np.ones([5, 5])
	delay = 1.

	max_num_paths = 1

	network = payment_network(graph, demand_mat, credit_mat, delay, max_num_paths)

	primal_values = np.zeros([1, NUM_ITERATIONS])
	dual_values = np.zeros([1, NUM_ITERATIONS])

	for step in range(NUM_ITERATIONS):
		network.update_flows()
		primal_values[0, step] = network.print_primal_value()
		network.update_prices()
		dual_values[0, step] = network.print_dual_value()

	print primal_values
	print dual_values

	np.save('./primal_values.npy', primal_values)	
	np.save('./dual_values.npy', dual_values)	

if __name__=='__main__':
	main()