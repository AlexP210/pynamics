"""
Module containing the definition of all available joints.
"""

import abc

import numpy as np
from pynamics.math.integrators import VerletIntegrator, ForwardEulerQuaternionIntegrator
from pynamics.constants import EPSILON


class Joint(abc.ABC):
    """Abstract base class defining the :code:`Joint` interface"""
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_translation_vector(self):
        """
        Get the translation vector across a joint, expressed in the child body \
        frame.
        """
        pass

    @abc.abstractmethod
    def get_rotation_matrix(self):
        """
        Get the rotation matrix across a joint, transforming vectors from the \
        child body frame to the parent body frame.
        """
        pass

    @abc.abstractmethod
    def get_c(self):
        """
        Get c_J for the joint, as defined on pg. 55 of Rigid Body Dynamics Algorithms.

        c_J = \opendot{S} \dot{q} + \opendot{\sigma}
        S = Joint Space to Cartesian Space Projection Matrix
        q = Joint Space Position
        \sigma = Bias Velocity

        This is zero for any joint with a constant S-matrix.
        """
        pass

    def get_motion_subspace(self):
        """
        The S matrix for the joint, converting joint-space velocity to cartesian \
        space velocity.

        v_J = S(q, t) \dot{q} + \sigma(q,t)

        """
        return self.motion_subspace

    def get_constraint_force_subspace(self):
        """
        The T matrix for the joint, converting the constraint sub-space force \
        to the cartesian space force across a joint.

        f_constraint = T\lambda

        f_constraint = The constraint force acting across the joint, in cartesian \
        space
        T = The Constraint Space
        \lambda = The constraint-space force
        """
        return self.constraint_force_subspace

    def get_velocity(self):
        """
        Get the velocity across the joint, v_J.
        
        v_child = v_parent + v_J
        """
        return self.get_motion_subspace() @ self.get_configuration_d()

    def get_acceleration(self, q_dd):
        """
        Get the acceleration across the joint, a_J.
        
        a_child = a_parent + a_Jq
        """
        return self.get_motion_subspace() @ q_dd

    def get_configuration(self):
        """
        Get the joint space position, q.
        """
        return self.q

    def get_configuration_d(self):
        """
        Get the joint space velocity \dot{q}.
        """
        return self.q_d

    def set_configuration(self, configuration):
        """
        Set the joint space position q.
        """
        self.q = configuration

    def set_configuration_d(self, configuration_d):
        """
        Set the joint space velocity, \dot{q}.
        """
        self.q_d = configuration_d

    def get_T(self):
        """
        Get the affine transform across the joint.
        """
        T = np.matrix(np.zeros(shape=(4, 4)))
        C = self.get_rotation_matrix()
        R = self.get_translation_vector()
        for i in range(3):
            for j in range(3):
                T[i, j] = C[i, j]
            T[i, 3] = R[i, 0]
        T[3, 3] = 1
        return T

    def integrate(self, dt:float, q_dd:np.matrix):
        """
        Use the integrator to increment the state (q) and velocity (\dot{q})

        Args:
            dt (float): Time step to increment the integrator.
            q_dd (np.matrix): Acceleration to integrate.
        """
        self.q, self.q_d = self.integrator.step(dt, q_dd)

    def initialize_integrator(self):
        """
        Initialize the state of the integrator with the current state of the \
        joint
        """
        self.integrator.initialize_state(self.q, self.q_d)


class RevoluteJoint(Joint):
    """
    A joint that revolves freely around an axis.
    
    .. note::

        The joint-space configuration q = [\theta,] represents the rotation angle,
        about the joint axis.

        The joint-space velocity \dot{q} = [\dot{\theta},] represents
        the rate of change of the rotation angle about the joint axis.
    
    """
    def __init__(self, axis: int):
        """
        Create a revolute joint.

        Args:
            axis (int): The index of axis of rotation (0 = x, 1 = y, 2 = z)
        """
        self.axis = axis
        self.motion_subspace = np.matrix(np.zeros((6, 1)))
        self.motion_subspace[axis] = 1
        self.constraint_force_subspace = np.matrix(np.zeros(shape=(6, 5)))

        temp = 0
        for i in range(6):
            if i == axis:
                continue
            self.constraint_force_subspace[i, temp] = 1
            temp += 1
        self.q = np.matrix(np.zeros(1)).T
        self.q_d = np.matrix(np.zeros(self.q.shape)).T
        self.integrator = VerletIntegrator()

    def get_translation_vector(self):
        r = np.matrix(np.zeros(shape=(3, 1)))
        return r

    def get_rotation_matrix(self):
        c = np.cos(self.q)[0, 0]
        s = np.sin(self.q)[0, 0]
        C = 1 - c
        rotation_axis = np.matrix(np.zeros(3)).T
        rotation_axis[self.axis] = 1
        x = rotation_axis[0, 0]
        y = rotation_axis[1, 0]
        z = rotation_axis[2, 0]
        return np.matrix(
            [
                [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
                [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
                [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
            ]
        )

    def get_c(self):
        return np.matrix(np.zeros(6)).T


class FreeJoint(Joint):
    """
    A joint which is not constrained on any axis.

    .. note::

        The joint-space configuration q = [q_w, q_x, q_y, q_z, r_x, r_y, r_z]
        represents the concatenation of the orientation quaternion from the
        parent frame to the child frame (with real component first), and the 
        parent->child translation vector, expressed in the child frame.

        The joint-space velocity \dot{q} = [w_x, w_y, w_z, v_x, v_y, v_z] 
        represents the spatial velocity across the joint, with angular component
        first.  
    
    """
    def __init__(self):
        """
        Create a free joint.
        """
        self.motion_subspace = np.matrix(np.eye(6, 6))
        self.constraint_force_subspace = np.matrix(np.zeros(shape=(6, 0)))
        self.q = np.matrix(np.zeros(shape=(7, 1)))
        self.q[0] = 1
        self.q_d = np.matrix(np.zeros(shape=(6, 1)))
        self.position_integrator = VerletIntegrator()
        self.orientation_integrator = ForwardEulerQuaternionIntegrator()

    def get_translation_vector(self):
        r = self.q[4:, 0]
        return self.get_rotation_matrix() @ r

    def get_rotation_matrix(self):
        rotation_angle = 2 * np.atan2(np.linalg.norm(self.q[1:4, 0]), self.q[0, 0])
        if rotation_angle < EPSILON:
            return np.matrix(np.eye(3, 3))
        rotation_axis = self.q[1:4, 0] / np.linalg.norm(self.q[1:4, 0])
        c = np.cos(rotation_angle)
        s = np.sin(rotation_angle)
        C = 1 - c
        x = rotation_axis[0, 0]
        y = rotation_axis[1, 0]
        z = rotation_axis[2, 0]
        return np.matrix(
            [
                [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
                [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
                [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
            ]
        )

    def get_c(self):
        return np.matrix(np.zeros(6)).T

    def initialize_integrator(self):
        orientation = self.q[:4]
        position = self.q[4:]
        orientation_d = self.q_d[:3]
        position_d = self.q_d[3:]
        self.position_integrator.initialize_state(position, position_d)
        self.orientation_integrator.initialize_state(orientation, orientation_d)

    def integrate(self, dt, q_dd):
        orientation_dd = q_dd[:3]
        position_dd = q_dd[3:]
        position, position_d = self.position_integrator.step(dt, position_dd)
        orientation, orientation_d = self.orientation_integrator.step(
            dt, orientation_dd
        )
        self.q = np.concatenate(
            (orientation.astype(float), position.astype(float)), axis=0
        )
        self.q_d = np.concatenate(
            (orientation_d.astype(float), position_d.astype(float)), axis=0
        )


class TranslationJoint(Joint):
    """
    A joint which can slide freely along any axis, but not rotate.
    
    .. note::

        The joint-space configuration q = [r_x, r_y, r_z] represents the x, y, z components
        of the parent->child frame translation, expressed in the child frame.

        The joint-space velocity \dot{q} = [v_x, v_y, v_z] represents
        the x, y, z components of the velocity of the child relative to the
        parent, expressed in the child frame.

    """
    def __init__(self):
        """
        Create a translational joint.
        """
        self.motion_subspace = np.matrix(
            np.block([[np.zeros(shape=(3, 3))], [np.eye(3, 3)]])
        )
        self.constraint_force_subspace = np.matrix(
            np.block([[np.eye(3, 3)], [np.zeros(shape=(3, 3))]])
        )
        self.q = np.matrix(np.zeros(shape=(3, 1)))
        self.q_d = np.matrix(np.zeros(self.q.shape))
        self.integrator = VerletIntegrator()

    def get_translation_vector(self):
        r = self.q
        return r

    def get_rotation_matrix(self):
        return np.matrix(np.eye(3, 3))

    def get_c(self):
        return np.matrix(np.zeros(6)).T


class FixedJoint(Joint):
    """
    A joint which is constrained on every axis.
    
    .. note::

        The joint-space configuration q = [] is empty, as there are no degrees of freedom.

        The joint-space velocity \dot{q} = [] is empty, as there are no degrees of freedom.
    
    """
    def __init__(self):
        """
        Create a fixed joint.
        """
        self.motion_subspace = np.matrix(np.zeros(shape=(6, 0)))
        self.constraint_force_subspace = np.matrix(np.eye(6, 6))
        self.q = np.matrix(np.zeros(shape=(0, 1)))
        self.q_d = np.matrix(np.zeros(self.q.shape))
        self.integrator = VerletIntegrator()

    def get_translation_vector(self):
        r = np.matrix(np.zeros(shape=(3, 1)))
        return r

    def get_rotation_matrix(self):
        return np.matrix(np.eye(3, 3))

    def get_c(self):
        return 0

    def get_velocity(self):
        return np.matrix(np.zeros(shape=(6, 1)))

    def get_acceleration(self, q_dd):
        return np.matrix(np.zeros(shape=(6, 1)))
