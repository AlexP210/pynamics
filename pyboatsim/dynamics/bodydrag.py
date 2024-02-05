import typing

import numpy as np
import scipy.integrate as integrate

from dynamics import DynamicsParent
from state import State

class BodyDrag(DynamicsParent):
    def __init__(
            self,
            cross_sectional_area: float,
            drag_coefficient: float,
        ):
        self.dynamics_parameters = {
            "cross_sectional_area": cross_sectional_area,
            "drag_coefficient": drag_coefficient,
        }
        self.name = f"bodydrag"

    def required_state_labels(self):
        return [
            "rho",
            "v_boat",
            "v_water"
        ]
    
    def compute_dynamics(self, state:State):
        factors = [
            np.sign(state["v_water"] - state["v_boat"]),
            0.5*self.dynamics_parameters["drag_coefficient"]*state["rho"],
            self.dynamics_parameters["cross_sectional_area"],
            (state["v_water"]-state["v_boat"])**2
        ]
        return np.prod(factors)

