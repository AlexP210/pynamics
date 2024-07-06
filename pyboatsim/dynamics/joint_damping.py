import typing

import numpy as np
import trimesh

from pyboatsim.dynamics import JointDynamicsParent
from pyboatsim.state import State
from pyboatsim.constants import AXES, EPSILON
from pyboatsim.kinematics.topology import Topology

from pyboatsim.math.linalg import R3_cross_product_matrix

class JointDamping(JointDynamicsParent):
    def __init__(
            self,
            name: str,
            damping_factor: float,
        ):
        super().__init__(name=name)
        self.damping_factor = damping_factor
    
    def compute_dynamics(self, topology:Topology, body_name:str) -> typing.Tuple[np.matrix, np.matrix]:
        return [-self.damping_factor * topology.joints[body_name].get_configuration_d()]
