import numpy as np

def cross_product_matrix(v:np.matrix):
    return np.matrix([
        [0, -v[2,0], v[1,0]],
        [v[2,0], 0, -v[0,0]],
        [-v[1,0], v[0,0], 0]
    ])