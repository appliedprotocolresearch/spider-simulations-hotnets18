""" algorithm to compute k shortest paths from given source to destination
in a graph using Yen's algorithm: https://en.wikipedia.org/wiki/Yen%27s_algorithm """

import copy 
import cPickle as pickle 
import networkx as nx 
import numpy as np

def ksp_yen(graph, node_start, node_end, max_k=2):

    graph = copy.deepcopy(graph)

    A = []
    B = []

    try:
        path = nx.shortest_path(graph, source=node_start, target=node_end)
    except:
        print "No path found!"
        return None 

    A.append(path)    
    
    for k in range(1, max_k):
        for i in range(0, len(A[-1])-1):

            node_spur = A[-1][i]
            path_root = A[-1][:i+1]

            edges_removed = []
            for path_k in A:
                curr_path = path_k
                if len(curr_path) > i and path_root == curr_path[:i+1]:
                    if (curr_path[i], curr_path[i+1]) in graph.edges() or \
                        (curr_path[i+1], curr_path[i]) in graph.edges():
                        graph.remove_edge(curr_path[i], curr_path[i+1])
                        edges_removed.append([curr_path[i], curr_path[i+1]])
            
            nodes_removed = []
            for rootpathnode in path_root:
                if rootpathnode != node_spur:
                    graph.remove_node(rootpathnode)
                    nodes_removed.append(rootpathnode)

            try:
                path_spur = nx.shortest_path(graph, source=node_spur, target=node_end)
            except:
                path_spur = None
            
            if path_spur:
                path_total = path_root[:-1] + path_spur            
                potential_k = path_total            
                if not (potential_k in B):
                    B.append(potential_k)
            
            for node in nodes_removed:
                graph.add_node(node)

            for edge in edges_removed:
                graph.add_edge(edge[0], edge[1])
        
        if len(B):
            B.sort(key=len)
            A.append(B[0])
            B.pop(0)
        else:
            break
    
    return A

def ksp_edge_disjoint(graph, node_start, node_end, max_k=2):
    """ compute k edge disjoint shortest paths """
    graph = copy.deepcopy(graph)

    A = []

    try:
        path = nx.shortest_path(graph, source=node_start, target=node_end)
    except:
        print "No path found!"
        return None 

    A.append(path)    
    
    for k in range(1, max_k):
        prev_path = A[-1]
        for i, j in zip(prev_path[:-1], prev_path[1:]):
            if (i, j) in graph.edges() or (j, i) in graph.edges():
                graph.remove_edge(i, j)

        try:
            path = nx.shortest_path(graph, source=node_start, target=node_end)
        except:
            path = None

        if path:
            A.append(path)
                
    return A    

def kwp_edge_disjoint(graph, node_start, node_end, max_k, credit_mat, delay):
    """ compute k edge disjoint widest paths """
    """ using http://www.cs.cmu.edu/~avrim/451f08/lectures/lect1007.pdf """

    graph = copy.deepcopy(graph)
    capacity_mat = credit_mat / delay
    A = []

    try:
        path = nx.shortest_path(graph, source=node_start, target=node_end)
    except:
        print "No path found!"
        return None 

    for k in range(max_k):
        widthto = {}
        pathto = {}
        tree_nodes = []
        tree_neighbors = []
        tree_nodes_membership_indicator = {v: False for v in graph.nodes()}
        tree_neighbors_membership_indicator = {v: False for v in graph.nodes()}
        
        widthto[node_end] = np.inf
        pathto[node_end] = None
        tree_nodes.append(node_end)
        tree_nodes_membership_indicator[node_end] = True
        tree_neighbors = graph.neighbors(node_end)
        for v in graph.neighbors(node_end):
            tree_neighbors_membership_indicator[v] = True

        while tree_neighbors and (tree_nodes_membership_indicator[node_start] is False):
            x = tree_neighbors.pop(0)
            tree_neighbors_membership_indicator[x] = False
            
            max_width = -1.
            max_width_neighbor = None            
            for v in graph.neighbors(x):
                if tree_nodes_membership_indicator[v] is True:
                    if np.min([widthto[v], capacity_mat[x, v]]) > max_width:
                        max_width = np.min([widthto[v], capacity_mat[x, v]])
                        max_width_neighbor = v
                else:
                    if tree_neighbors_membership_indicator[v] is False:
                        tree_neighbors.append(v)
                        tree_neighbors_membership_indicator[v] = True

            widthto[x] = max_width
            pathto[x] = max_width_neighbor
            tree_nodes.append(x)
            tree_nodes_membership_indicator[x] = True

        if tree_nodes_membership_indicator[node_start] is True:
            node = node_start
            path = [node]
            while node != node_end:
                node = pathto[node]
                path.append(node)
            A.append(path)

        prev_path = A[-1]
        for i, j in zip(prev_path[:-1], prev_path[1:]):
            if (i, j) in graph.edges() or (j, i) in graph.edges():
                graph.remove_edge(i, j)            

    return A 

def raeke(node_start, node_end):
    with open('../oblivious_routing/raeke_0.pkl', 'rb') as input:
        paths = pickle.load(input)

    return paths[node_start, node_end]

def main():

    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2, 3])    
    graph.add_edges_from([(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)])

    credit_mat = np.zeros([4, 4])
    credit_mat[0, 1] = 1.
    credit_mat[1, 2] = 3.
    credit_mat[2, 3] = 2.
    credit_mat[0, 2] = 2.
    credit_mat[0, 3] = 5.
    delay = 1.

    print kwp_edge_disjoint(graph, 0, 3, 2, credit_mat, delay)

if __name__=='__main__':
    main()
