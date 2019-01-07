""" gurobi time limit """
TIME_LIMIT = 3600

""" graph size """
GRAPH_SIZE = 30

""" graph types: isp, scale_free, small_world, erdos_renyi, test """
GRAPH_TYPE = 'scale_free'

""" demand matrix types: uniform, skew, test """
SRC_TYPE = 'uniform'

""" skew rate: real number > 0 """
SKEW_RATE = 0.25

""" credit type: uniform, random """
CREDIT_TYPE = 'uniform'

""" number of paths to consider """
MAX_NUM_PATHS = 4

""" type of path: ksp, ksp_edge_disjoint, kwp_edge_disjoint, raeke """
PATH_TYPE = 'raeke'

""" amount of credits on edges """
CREDIT_AMT = 1.0

""" transaction value """
TXN_VALUE = 1./1000

""" delay """
DELAY = 1.

""" random seed """
RAND_SEED = 11