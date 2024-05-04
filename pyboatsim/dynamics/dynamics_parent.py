import typing
import abc

from pyboatsim.state import State
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
    def compute_dynamics(self, state:State, topology:Topology, dt:float) -> State:
        """
        Each instance of Dynamics needs to implement a calculation of the 
        dynamics, adding a state
        """
        raise NotImplementedError(
            "Implement `compute_dynamics()` in your `Dynamics` subclass."
            )
    
    @abc.abstractmethod
    def required_state_labels(self):
        """
        Each instance of Dynamics needs to implement a function stating what
        labels need to exist in the state dictionary in order for it to work
        """
        raise NotImplementedError(
            "Implement `required_state_labels()` in your `Dynamics` subclass."
            )


    def __call__(self, state:State, topology:Topology, dt:float) -> float:
        """
        Handles checking if the passed `State` object contains all the required
        labels, printing a helpful error message if not, and calculating 
        dynamics afterwards.
        """
        missing_labels = [
            label 
            for label in self.required_state_labels() if not label in state.labels()
        ]
        if len(missing_labels) != 0: raise ValueError(
            f"The following labels are missing from the sim state, dynamics"
            f" cannot be calculated: {', '.join(missing_labels)}")
        return self.compute_dynamics(state, topology, dt)