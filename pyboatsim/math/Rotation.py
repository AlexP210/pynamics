import numpy as np

from pyboatsim.constants import EPSILON

class Rotation:

    def __init__(self, axis_angle=None, rotation_matrix=None, quaternion=None, xyz_euler=None):
        number_of_provided_args = sum([arg is not None for arg in (axis_angle, rotation_matrix, quaternion, xyz_euler)])
        if number_of_provided_args < 1:
            raise ValueError("Must provide a rotation definition when instantiating a Rotation")
        elif number_of_provided_args > 1:
            raise ValueError("Cannot provide more than one rotation definition when instantiating a Rotation")

        if axis_angle is not None:
            angle = np.linalg.norm(axis_angle)
            axis = axis_angle / angle

            c = np.cos(angle)
            s = np.sin(angle)
            C = 1-c
            x = axis[0,0]
            y = axis[1,0]
            z = axis[2,0]

            self.rotation_matrix = np.matrix([
                [x*x*C+c, x*y*C-z*s, x*z*C+y*s],
                [y*x*C+z*s, y*y*C+c, y*z*C-x*s],
                [z*x*C-y*s, z*y*C+x*s, z*z*C+c]
            ])
        
        elif rotation_matrix is not None:
            if abs((rotation_matrix.T @ rotation_matrix - np.matrix(np.eye(3,3))).trace() - 3) > EPSILON:
                raise ValueError("Provided rotation matrix is not orthogonal")
            self.rotation_matrix = rotation_matrix

        elif quaternion is not None or xyz_euler is not None:
            raise ValueError("Quaternion and xyz_euler are not yet implemented.")
