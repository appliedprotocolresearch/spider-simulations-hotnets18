""" gurobi time limit """
TIME_LIMIT = 3600

""" graph size """
GRAPH_SIZE = 102

""" graph types: test, scale_free, small_world, erdos_renyi, isp, lnd """
GRAPH_TYPE = 'lnd'

""" demand matrix types: test, uniform, skew, lnd """
SRC_TYPE = 'lnd'

""" skew rate: real number > 0 """
SKEW_RATE = 0.25

""" credit type: uniform, random, lnd """
CREDIT_TYPE = 'uniform'

""" number of paths to consider """
MAX_NUM_PATHS = 4

""" type of path: ksp, ksp_edge_disjoint, kwp_edge_disjoint, raeke """
PATH_TYPE = 'ksp_edge_disjoint'

""" type of LP: maxdelta, pathdelta """
LP_TYPE = 'pathdelta'

""" transaction value """
TXN_VALUE = 1./1000

""" single-hop delay (in seconds) """
SINGLE_HOP_DELAY = 0.03

""" random seed """
RAND_SEED = 11