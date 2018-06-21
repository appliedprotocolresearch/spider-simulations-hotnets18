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
