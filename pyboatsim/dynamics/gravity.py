import typing

import numpy as np
import trimesh

from pyboatsim.dynamics import DynamicsParent
from pyboatsim.state import State
from pyboatsim.constants import AXES, EPSILON
from pyboatsim.kinematics.topology import Topology

class Gravity(DynamicsParent):
    def __init__(
            self,
            name: str,
            g: float,
            direction: int
        ):
        super().__init__(name=name)
        self.g = g
        self.direction = direction
    
    def compute_dynamics(self, topology:Topology, body_name:str) -> State:
        d = np.matrix(np.zeros(shape=(6,1)))
        d[self.direction, 0] = 1
        return topology.bodies[body_name].mass * self.g * d
