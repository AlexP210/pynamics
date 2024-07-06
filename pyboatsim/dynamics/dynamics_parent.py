import typing
import abc
import numpy as np

from pyboatsim.kinematics.topology import Topology
from pyboatsim.math.linalg import R3_cross_product_matrix

class BodyDynamicsParent(abc.ABC):
    def __init__(self, name):
        """
        Each instance of Dynamics needs to implement an initialization that
        creates an attribute `dynamics_parameters`, which is a dictionary
        that contains the parameters used by the dynamics module.
        """
        self.name = name

    @abc.abstractmethod
    def compute_dynamics(self, topology:Topology, body_name:str) -> typing.List[typing.Tuple[np.matrix, np.matrix]]:
        """
        Each instance of Dynamics needs to implement a calculation of the 
        dynamics, to compute the 6-D force/moment vector given a topology and body_name
        """
        raise NotImplementedError(
            "Implement `compute_dynamics()` in your `Dynamics` subclass."
            )

    def __call__(self, topology:Topology, body_name:str) -> np.matrix:
        total_wrench = np.matrix(np.zeros(shape=(6,1)))

        for force, point_of_application in self.compute_dynamics(topology, body_name):
            wrench = np.matrix(np.zeros(shape=(6,1)))
            wrench[:3,0] = R3_cross_product_matrix(point_of_application) @ force
            wrench[3:,0] = force
            total_wrench += wrench
        return wrench
    
class JointDynamicsParent(abc.ABC):
    def __init__(self, name):
        """
        Each instance of Dynamics needs to implement an initialization that
        creates an attribute `dynamics_parameters`, which is a dictionary
        that contains the parameters used by the dynamics module.
        """
        self.name = name

    @abc.abstractmethod
    def compute_dynamics(self, topology:Topology, body_name:str) -> np.matrix:
        """
        Each instance of Dynamics needs to implement a calculation of the 
        dynamics, to return a joint-space force given a topology and body_name
        """
        raise NotImplementedError(
            "Implement `compute_dynamics()` in your `Dynamics` subclass."
            )

    def __call__(self, topology:Topology, body_name:str) -> np.matrix:    
        return sum(self.compute_dynamics(topology, body_name))