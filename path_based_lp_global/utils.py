""" gurobi time limit """
TIME_LIMIT = 3600

""" graph size """
GRAPH_SIZE = 36

""" graph types: isp, scale_free, erdos_renyi, test """
GRAPH_TYPE = 'scale_free'

""" source types: uniform, skew, test """
SRC_TYPE = 'uniform'

""" skew rate: real number > 0 """
SKEW_RATE = 0.25

""" credit type: uniform, random """
CREDIT_TYPE = 'random'

""" number of paths to consider """
MAX_NUM_PATHS = 1

""" type of path: ksp, ksp_edge_disjoint, kwp_edge_disjoint """
PATH_TYPE = 'kwp_edge_disjoint'

""" amount of credits on edges """
CREDIT_AMT = 1.0

""" transaction value """
TXN_VALUE = 1./1000

""" delay """
DELAY = 1.

""" random seed """
RAND_SEED = 11