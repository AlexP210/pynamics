import typing

import numpy as np
import scipy.integrate as integrate

from pyboatsim.dynamics import DynamicsParent
from pyboatsim.state import State
from pyboatsim.constants import AXES

class ConstantForce(DynamicsParent):
    def __init__(
            self,
            name: str,
            force_x: float = 0,
            force_y: float = 0,
            force_z: float = 0,
            tau_x: float = 0,
            tau_y: float = 0,
            tau_z: float = 0,
        ):
        super().__init__(name=name)
        self.dynamics_parameters = {
            "f_x": force_x,
            "f_y": force_y,
            "f_z": force_z,
            "tau_x": tau_x,
            "tau_y": tau_y,
            "tau_z": tau_z
        }

    def required_state_labels(self):
        return []
    
    def compute_dynamics(self, state:State, dt:float) -> State:
        for axis in AXES:
            state.set({
                f"f_{axis}__{self.name}": self.dynamics_parameters[f"f_{axis}"],
                f"tau_{axis}__{self.name}": self.dynamics_parameters[f"tau_{axis}"]
            })

<<<<<<< HEAD
        return state
=======
        return state

>>>>>>> develop
