import typing

import numpy as np
import scipy.integrate as integrate

from pyboatsim.dynamics import DynamicsParent
from pyboatsim.state import State
from pyboatsim.constants import AXES

class SimpleBodyDrag(DynamicsParent):
    def __init__(
            self,
            name: str,
            cross_sectional_area: float,
            drag_coefficient: float,
        ):
        super().__init__(name=name)
        self.dynamics_parameters = {
            "cross_sectional_area": cross_sectional_area,
            "drag_coefficient": drag_coefficient,
        }
        self.name = f"bodydrag"

    def required_state_labels(self):
        return [
            "rho" 
            ] + [
                f"v_{axis}__boat" for axis in AXES
            ] + [
                f"v_{axis}__water" for axis in AXES
            ]
    
    def compute_dynamics(self, state:State, dt:float) -> State:
        for axis in AXES:
            factors = [
                np.sign(state[f"v_{axis}__water"] - state[f"v_{axis}__boat"]),
                0.5*self.dynamics_parameters["drag_coefficient"]*state["rho"],
                self.dynamics_parameters["cross_sectional_area"],
                (state[f"v_{axis}__water"] - state[f"v_{axis}__boat"])**2
            ]
            state.set({f"f_{axis}__{self.name}": np.prod(factors)})
            state.set({f"tau_{axis}__{self.name}": 0})
        return state

class BodyDrag(DynamicsParent):
    def __init__(self, model_path:str):
        pass