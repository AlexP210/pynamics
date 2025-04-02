"""Linear Algebra"""
import numpy as np

def R3_cross_product_matrix(v:np.matrix):
    return np.matrix([
        [0, -v[2,0], v[1,0]],
        [v[2,0], 0, -v[0,0]],
        [-v[1,0], v[0,0], 0]
    ])

def cross(v:np.matrix):
    top_left = R3_cross_product_matrix(v[0:3])
    top_right = np.matrix(np.zeros(shape=(3,3)))
    bottom_left = R3_cross_product_matrix(v[3:])
    bottom_right = R3_cross_product_matrix(v[0:3])
    return np.matrix(np.block([
        [top_left, top_right],
        [bottom_left, bottom_right]
    ]))

def cross_star(v:np.matrix):
    return -cross(v).T

def matrix_exponential(A:np.matrix) -> np.matrix:
    """
    Calculate the matrix exponential.

    Args:
        A (np.matrix): A matrix for which to calculate the matrix exponential.

    Returns:
        np.matrix: The matrix exponential of A
    """
    eigen_values, eigen_vectors = np.linalg.eig(A)
    return eigen_vectors * np.diag(np.exp(eigen_values)) * np.linalg.inv(eigen_vectors)