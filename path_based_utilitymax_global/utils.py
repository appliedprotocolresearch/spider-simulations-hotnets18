""" graph size """
GRAPH_SIZE = 10

""" graph types: isp, scale_free, small_world, test """
GRAPH_TYPE = 'small_world'

""" source types: uniform, skew, test, pickle """
SRC_TYPE = 'pickle'

""" skew rate: real number > 0 """
SKEW_RATE = 0.25

""" credit type: uniform """
CREDIT_TYPE = 'uniform'

""" number of paths to consider """
MAX_NUM_PATHS = 4

""" type of path: ksp, ksp_edge_disjoint, kwp_edge_disjoint """
PATH_TYPE = 'ksp_edge_disjoint'

""" transaction value """
TXN_VALUE = 1./1000

""" type of LP: maxdelta, pathdelta """
LP_TYPE = 'pathdelta'

""" single-hop delay (in seconds) """
SINGLE_HOP_DELAY = 0.03

""" random seed """
RAND_SEED = 23