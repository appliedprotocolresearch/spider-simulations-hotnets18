""" gurobi time limit """
TIME_LIMIT = 3600

""" graph size """
GRAPH_SIZE = 20

""" graph types: isp, scale_free, ripple """
GRAPH_TYPE = 'ripple'

""" source types: uniform, skew, ripple """
SRC_TYPE = 'ripple'

""" skew rate: real number > 0 """
SKEW_RATE = 0.25

""" credit type: uniform, ripple """
CREDIT_TYPE = 'ripple'

""" amount of credits on edges """
CREDIT_AMT = 250.

""" transaction value """
TXN_VALUE = 1.

""" ripple credit link file """
RIPPLE_CREDIT_PATH = '../RippleStaticClean_0.0.graph_CREDIT_LINKS'

""" ripple transaction dataset file """
RIPPLE_TXN_PATH = '../RippleStaticClean_45000_Tr.txt'