""" program to simulate distributed flow and price update operations, 
and analyze their convergence behavior """

import copy
import networkx as nx 
import numpy as np 

from utils import *

class payment_network(object):
	def __init__(self, graph, demand_mat, max_num_paths):
		""" initialize graph """
		self.graph = copy.deepcopy(graph)
		self.demand_mat = demand_mat

		assert np.shape(demand_mat)[0] == np.shape(demand_mat)[1]
		assert len(graph.nodes()) = np.shape(demand_mat)[0]

		""" initialize flows and prices on links """
		self.link_flows = {}
		self.link_prices = {}

		for e in self.graph.edges():
			self.link_flows[(e[0], e[1])] = 0.
			self.link_flows[(e[1], e[0])] = 0.
			self.link_prices[(e[0], e[1])] = 1.
			self.link_prices[(e[0], e[1])] = 1.

		""" initalize paths for each source/destination pair """
		self.path_select(max_num_paths)

	def path_select(self, max_num_paths):

		if max_num_paths == np.inf:
			return
		else:
			self.paths = {}

		compute and store at most k paths for each source, destination pair of nodes
		e.g., k shortest paths

	def flow_update(self):
		depending on link prices update flow variables 
		update graph state

		if max_num_paths is infinity, then update flow variables only on one shortest and longest path

	def price_update(self):
		depending on link utilization update price variables on all links 
		update graph state

def main():
	load ISP graph 
	initialize payment_network

	for a number of iterations do 
		flow_update
		price_update

	print final answer

if __name__=='__main__':
	main()