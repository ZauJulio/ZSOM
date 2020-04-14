import numpy as np 

# def find_bmu_act(vec, act):
#     """  """
#     for node in act:
#         for data in node[2]:
#             if vec in data:
#                 return node

# def dist(a, b):
#     """ Return euclidean distance """
#     return np.sqrt(np.sum(np.square(np.subtract(a, b))))

""" Error metric per node """

def qE_node(act):
    """ Return list of mean quantization error by node """
    return [np.mean([qE(data, node[1]) for data in node[2]]) for node in act]

def MAE_node(act):
    """ Return list of mean MAE by node """
    return [np.mean([MAE(data, node[1]) for data in node[2]]) for node in act]

def MSE_node(act):
    """ Return list of mean MSE by node """
    return [np.mean([MSE(data, node[1]) for data in node[2]]) for node in act]

def RMSE_node(act):
    """ Return list of mean RMSE by node """
    return [np.mean([RMSE(data, node[1]) for data in node[2]]) for node in act]

def MAPE_node(act):
    """ Return list of mean MAPE by node """
    return [np.mean([MAPE(data, node[1]) for data in node[2]]) for node in act]

""" Error metrics """

def qE(a, b):
    """ Return quantization error """
    return np.sqrt(np.sum(np.square(np.subtract(a, b))))

def MAE(a, b):
    """ Return Mean absolute error """
    return abs(np.subtract(a, b)).mean()

def MSE(a, b):
    """ Return Mean squared error """
    return np.square(np.subtract(a, b)).mean()

def RMSE(a, b):
    """ Return Root mean squared error """
    return np.sqrt(np.square(np.subtract(a, b)).mean())

def MAPE(a, b):
    """ Return Mean absolute percentage error """
    return abs(np.divide(np.subtract(a, b),a))*100
