import numpy as np
from scipy.sparse import lil_matrix

# parses a .gv file
def parse_graph_data(filename):
	count = 0
	adjacency_dict = {}
	with open(filename) as f:
		for line in f:
			count += 1
			parts = line.lstrip('\t').rstrip(';\n') .split(" ")
			if count == 1 and parts[0] != 'graph':
				raise ValueError("incorrect file format")
			if (len(parts) == 3 and parts[0].isdigit() and parts[2].isdigit()): #undirected edge
				src = int(parts[0])
				dst = int(parts[2])
				src_adj = adjacency_dict.get(src, set([]))
				src_adj.add(dst)
				adjacency_dict[src] = src_adj
				
				dst_adj = adjacency_dict.get(dst, set([]))
				dst_adj.add(src)
				adjacency_dict[dst] = dst_adj
	return adjacency_dict

def convert_adj_dict_to_list(adj_dict):
	nodes = adj_dict.keys()
	edges = []
	for key in adj_dict.keys():
		for node in adj_dict[key]:
			edges.append((key, node))
	return nodes, edges

def get_graph(filename):
	adj_dict = parse_graph_data(filename)
	nodes, edges = convert_adj_dict_to_list(adj_dict)    
	return nodes, edges

# return adjacency dict as parsed from a credit link graph
def parse_credit_link_graph(filename):
	f = open(filename, "r")
	line_num = 0
	degree = {}
	adjacent = {}
	credits = {}
	for line in f:
		parts = line.split(" ")
		if len(parts) == 5:
			src = int(parts[0])
			dst = int(parts[1])
			credit = float(parts[4]) - float(parts[2])
			
			degree[src] = degree.get(src, 0) + 1
			degree[dst] = degree.get(dst, 0) + 1

			src_adj = adjacent.get(src, set([]))
			src_adj.add(dst)
			adjacent[src] = src_adj

			dst_adj = adjacent.get(dst, set([]))
			dst_adj.add(src)
			adjacent[dst] = dst_adj

			credits[src, dst] = credit

	return adjacent, credits

def convert_credit_dict_to_mat(credit_dict, n):
	credit_mat = lil_matrix((n, n))
	# credit_mat = np.zeros([n, n])
	for u, v in credit_dict.keys():
		credit_mat[u, v] = credit_dict[(u, v)]
		credit_mat[v, u] = credit_dict[(u, v)]
	return credit_mat

def read_demand_from_file(demand_file, num_nodes):
	demand_mat = lil_matrix((num_nodes, num_nodes))
	# demand_mat = np.zeros([num_nodes, num_nodes])
	count = 0
	with open(demand_file) as f:
		for line in f:
			parts = line.split(" ")
			src = int(parts[2])
			dst = int(parts[3])
			val = float(parts[1])
			demand_mat[src, dst] += val
			count += 1
	return demand_mat, count

if __name__=='__main__':
	filename = '../RippleStatic_0.0.graph_CREDIT_LINKS'
	adjacent, credits = parse_credit_link_graph(filename)

	nodes, edges = convert_adj_dict_to_list(adjacent)
	n = len(nodes)
	credit_mat = convert_credit_dict_to_mat(credits, n)

	demand_mat, count = read_demand_from_file('../RippleStaticClean_45000_Tr.txt', n)	
	print count
	print np.sum(demand_mat)

	# data = np.log10(np.array(credits.values()) + 1e-20)
	# pl.hist(data, 20) # , bins=np.logspace(np.log10(0.1),np.log10(7.0), 20))
	# pl.gca().set_xscale("log")
	# pl.show()

