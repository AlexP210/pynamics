import abc

import numpy as np
import quaternion
import pyboatsim.math.linalg as linalg
from pyboatsim.constants import EPSILON

class Joint(abc.ABC):
    
    @abc.abstractmethod
    def __init__(self): pass
    @abc.abstractmethod
    def get_translation_vector(self): pass
    @abc.abstractmethod
    def get_rotation_matrix(self): pass
    @abc.abstractmethod
    def get_c(self): pass
    
    def get_motion_subspace(self):
        return self.motion_subspace
    def get_constraint_force_subspace(self):
        return self.constraint_force_subspace
    def get_velocity(self):
        return self.get_motion_subspace()@self.get_configuration_d()
    def get_acceleration(self, q_dd):
        return self.get_motion_subspace()@q_dd
    def get_configuration(self):
        return self.q
    def get_configuration_d(self):
        return self.q_d
    def set_configuration(self, configuration):
        self.q = configuration
    def set_configuration_d(self, configuration_d):
        self.q_d = configuration_d
    def get_T(self):
        T = np.matrix(np.zeros(shape=(4,4)))
        C = self.get_rotation_matrix()
        R = self.get_translation_vector()
        for i in range(3):
            for j in range(3):
                T[i,j] = C[i,j]
            T[i,3] = R[i,0]
        T[3,3] = 1
        return T

    def integrate(self, dt, q_dd):
        self.q += (self.q_d + 0.5*q_dd*dt)*dt
        self.q_d += q_dd*dt

class RevoluteJoint(Joint):
    def __init__(self, axis:int):
        self.axis = axis
        self.motion_subspace = np.matrix(np.zeros((6,1)))
        self.motion_subspace[axis] = 1
        self.constraint_force_subspace = np.matrix(np.zeros(shape=(6,5)))

        temp = 0
        for i in range(6):
            if i == axis: continue 
            self.constraint_force_subspace[i,temp] = 1
            temp += 1
        self.q = np.matrix(np.zeros(1)).T
        self.q_d = np.matrix(np.zeros(self.q.shape)).T
        
    def get_translation_vector(self):
        r = np.matrix(np.zeros(shape=(3,1)))
        return r
    def get_rotation_matrix(self):
        c = np.cos(self.q)[0,0]
        s = np.sin(self.q)[0,0]
        C = 1-c
        rotation_axis = np.matrix(np.zeros(3)).T
        rotation_axis[self.axis] = 1
        x = rotation_axis[0,0]
        y = rotation_axis[1,0]
        z = rotation_axis[2,0]
        return np.matrix([
            [x*x*C+c, x*y*C-z*s, x*z*C+y*s],
            [y*x*C+z*s, y*y*C+c, y*z*C-x*s],
            [z*x*C-y*s, z*y*C+x*s, z*z*C+c]
        ])
    
    def get_c(self):
        return np.matrix(np.zeros(6)).T

class FreeJoint(Joint):
    def __init__(self):
        self.motion_subspace = np.matrix(np.eye(6,6))
        self.constraint_force_subspace = np.matrix(np.zeros(shape=(6,0)))
        self.q = np.matrix(np.zeros(shape=(7,1)))
        self.q_d = np.matrix(np.zeros(shape=(6,1)))

    def get_translation_vector(self):
        r = self.q[4:,0]
        return self.get_rotation_matrix()@r
    def get_rotation_matrix(self):
        rotation_angle = 2*np.atan2(np.linalg.norm(self.q[1:4,0]), self.q[0,0])
        if rotation_angle < EPSILON: return np.matrix(np.eye(3,3))
        rotation_axis = self.q[1:4,0] / np.linalg.norm(self.q[1:4,0])
        c = np.cos(rotation_angle)
        s = np.sin(rotation_angle)
        C = 1-c
        x = rotation_axis[0,0]
        y = rotation_axis[1,0]
        z = rotation_axis[2,0]
        return np.matrix([
            [x*x*C+c, x*y*C-z*s, x*z*C+y*s],
            [y*x*C+z*s, y*y*C+c, y*z*C-x*s],
            [z*x*C-y*s, z*y*C+x*s, z*z*C+c]
        ])
    def get_c(self):
        return np.matrix(np.zeros(6)).T
    def integrate(self, dt, q_dd):
        # q_d & q_dd is 6 elements
        new_q_d = self.q_d + q_dd * dt
        average_q_d = 0.5*(self.q_d + new_q_d)
        delta_rotation = quaternion.from_rotation_vector(average_q_d[:3,0].T*dt)
        current_rotation = quaternion.from_float_array(self.q[:4,0].T)
        new_rotation = quaternion.as_float_array(delta_rotation*current_rotation).T
        new_position = self.q[4:,0] + average_q_d[3:,0]*dt
        new_q = np.concat(
            (new_rotation.astype(float), new_position.astype(float)),
            axis=0
        )
        self.q_d = new_q_d
        self.q = new_q



class TranslationJoint(Joint):
    def __init__(self):
        self.motion_subspace = np.matrix(np.block(
            [[np.zeros(shape=(3,3))],
             [np.eye(3,3)]]
        ))
        self.constraint_force_subspace = np.matrix(np.block(
            [[np.eye(3,3)],
             [np.zeros(shape=(3,3))]]
        ))
        self.q = np.matrix(np.zeros(shape=(3,1)))
        self.q_d = np.matrix(np.zeros(self.q.shape))
    def get_translation_vector(self):
        r = self.q
        return r
    def get_rotation_matrix(self):
        return np.matrix(np.eye(3,3))
    def get_c(self):
        return np.matrix(np.zeros(6)).T


class FixedJoint(Joint):
    def __init__(self):
        self.motion_subspace = np.matrix(np.zeros(shape=(6,0)))
        self.constraint_force_subspace = np.matrix(np.eye(6,6))
        self.q = np.matrix(np.zeros(shape=(0,1)))
        self.q_d = np.matrix(np.zeros(self.q.shape))
        
    def get_translation_vector(self):
        r = np.matrix(np.zeros(shape=(3,1)))
        return r

    def get_rotation_matrix(self):
        return np.matrix(np.eye(3,3))
    def get_c(self):
        return 0
    def get_velocity(self):
        return np.matrix(np.zeros(shape=(6,1)))
    def get_acceleration(self, q_dd):
        return np.matrix(np.zeros(shape=(6,1)))