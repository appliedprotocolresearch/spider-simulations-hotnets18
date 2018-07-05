""" algorithm to compute k shortest paths from given source to destination
in a graph using Yen's algorithm: https://en.wikipedia.org/wiki/Yen%27s_algorithm """

import networkx as nx 

def ksp_yen(graph, node_start, node_end, max_k=2):

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

def main():
    
    graph = nx.fast_gnp_random_graph(30, 0.5)
    print graph.edges()
    print ksp_yen(graph, 0, 1, 3)

if __name__=='__main__':
    main()
