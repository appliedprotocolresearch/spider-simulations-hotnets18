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
		assert len(graph.nodes()) = np.shape(demand_mat)[0]

		""" initialize flows and prices on links """
		self.link_flows = {}
		self.link_prices_l = {}
		self.link_prices_y = {}
		self.link_prices_z = {}

		for i, j in self.nonzero_demands:
			self.link_prices_l[i, j] = 1.

		for e in self.graph.edges():
			self.link_flows[e[0], e[1]] = 0.
			self.link_flows[e[1], e[0]] = 0.
			self.link_prices_y[e[0], e[1]] = 1.
			self.link_prices_y[e[0], e[1]] = 1.
			self.link_prices_z[e[0], e[1]] = 1.
			self.link_prices_z[e[0], e[1]] = 1.			

		""" initalize path flows for each source/destination pair """
		self.path_flows = {}
		self.paths = self.preselect_paths(max_num_paths)

		for i, j in self.nonzero_demands:
			self.path_flows[i, j] = {}
			for path in self.paths[i, j]:
				self.path_flows[i, j][path] = 0.

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
		""" depending on link prices update flow variables """
		for i, j in self.nonzero_demands:
			for path in self.paths[i, j]:
				price = self.compute_path_price(path)
				temp_flow = self.path_flows[i, j][path]
				self.path_flows[i, j][path] += STEP_SIZE * (1. - price) # TODO: change precise update rule later
				self.path_flows[i, j][path] = np.max([0., self.path_flows[i, j][path]])

				""" update link flow states """
				for u, v in zip(path[:-1], path[1:]):
					self.link_flows[u, v] -= temp_flow
					self.link_flows[u, v] += self.path_flows[i, j][path]
		# TODO: if max_num_paths is infinity, then update flow variables only on one shortest and longest path

	def update_price(self):
		depending on link utilization update price variables on all links 
		update graph state

def main():
	load ISP graph 
	initialize payment_network

	for a number of iterations do 
		update_flows
		update_price

	print final answer

if __name__=='__main__':
	main()