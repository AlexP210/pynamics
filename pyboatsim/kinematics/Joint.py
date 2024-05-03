import abc

import numpy as np

import pyboatsim.math.linalg as lin

class Joint(abc.ABC):
    
    @abc.abstractmethod
    def __init__(self): pass

    @abc.abstractmethod
    def get_configuration(self): pass
    @abc.abstractmethod
    def get_configuration_d(self): pass
    @abc.abstractmethod
    def set_configuration(self, configuration): pass
    @abc.abstractmethod
    def set_configuration_d(self, configuration): pass


    @abc.abstractmethod
    def get_number_degrees_of_freedom(self): pass

    @abc.abstractmethod
    def get_translation_vector(self): pass
    @abc.abstractmethod
    def get_rotation_matrix(self): pass
    @abc.abstractmethod
    def get_T(self): pass
    @abc.abstractmethod
    def get_X(self): pass
    @abc.abstractmethod
    def get_Xstar(self): pass

    @abc.abstractmethod
    def get_motion_subspace(self): pass

    @abc.abstractmethod
    def get_constraint_force_subspace(self): pass

    @abc.abstractmethod
    def get_velocity(self): pass

    @abc.abstractmethod
    def get_c(self): pass

class RevoluteJoint(Joint):

    def __init__(self, axis:int):
        self.axis = axis
        self.motion_subspace = np.matrix(np.zeros(6)).T
        self.motion_subspace[axis] = 1
        self.constraint_force_subspace = np.matrix(np.zeros(shape=(6,5)))

        temp = 0
        for i in range(6):
            if i == axis: continue 
            self.constraint_force_subspace[i,temp] = 1
            temp += 1
        self.q = np.matrix(np.zeros(1)).T
        self.q_d = np.matrix(np.zeros(self.q.shape)).T

    def get_configuration(self):
        return self.q
    def get_configuration_d(self):
        return self.q_d
    def set_configuration(self, configuration):
        self.q = configuration
    def set_configuration_d(self, configuration_d):
        self.q_d = configuration_d
    
    def get_number_degrees_of_freedom(self):
        return 1
    
    def get_translation_vector(self):
        r = np.matrix(np.zeros(shape=(4,4))).T
        r[3,3] = 1
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
    
    def get_X(self):
        E = self.get_rotation_matrix()
        r = self.get_translation_vector()
        r_cross = lin.cross_product_matrix(r)
        return np.block([
            [E, np.zeros((3,3))],
            [-E@r_cross, E]
        ])
    def get_Xstar(self):
        E = self.get_rotation_matrix()
        r = self.get_translation_vector()
        r_cross = lin.cross_product_matrix(r)
        return np.block(
            [E, -E@r_cross],
            [np.zeros((3,3)), E]
        )


    def get_motion_subspace(self):
        return self.motion_subspace
    def get_constraint_force_subspace(self):
        return self.constraint_force_subspace


    def get_velocity(self):
        return self.get_motion_subspace()@self.get_configuration_d()
    
    def get_c(self):
        return np.matrix(np.zeros(6)).T
    
if __name__ == "__main__":
    joint = RevoluteJoint(axis=0)
    joint.set_configuration(np.matrix([[1]]))
    joint.set_configuration_d(np.matrix([[1]]))
    print(joint.get_velocity())