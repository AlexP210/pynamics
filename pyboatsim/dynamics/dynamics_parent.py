import typing
import abc
import numpy as np

from pyboatsim.kinematics.topology import Topology

class DynamicsParent(abc.ABC):
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
        dynamics, to compute the 6-D force/moment vector given a topology and body_name
        """
        raise NotImplementedError(
            "Implement `compute_dynamics()` in your `Dynamics` subclass."
            )

    def __call__(self, topology:Topology, body_name:str) -> float:
        return self.compute_dynamics(topology, body_name)