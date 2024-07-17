import typing

import numpy as np
import trimesh

from pyboatsim.dynamics import BodyDynamicsParent
from pyboatsim.state import State
from pyboatsim.constants import AXES, EPSILON
from pyboatsim.kinematics.topology import Topology

from pyboatsim.math.linalg import R3_cross_product_matrix

class Gravity(BodyDynamicsParent):
    def __init__(
            self,
            name: str,
            g: float,
            direction: int,
            body_names: typing.List[str] = []
        ):
        super().__init__(name=name, body_names=body_names)
        self.g = g
        self.direction = direction
    
    def compute_dynamics(self, topology:Topology, body_name:str) -> typing.Tuple[np.matrix, np.matrix]:
        force = np.matrix(np.zeros(shape=(3,1)))
        force[self.direction,0] = 1
        force *= self.g * topology.bodies[body_name].mass

        point_of_application = topology.get_transform("World", "Identity", body_name, "Center of Mass")[:3,3]
        return [(force, point_of_application),]
