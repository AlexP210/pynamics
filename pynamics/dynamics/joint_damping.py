import typing

import numpy as np
import trimesh

from pynamics.dynamics import JointDynamicsParent
from pynamics.state import State
from pynamics.constants import AXES, EPSILON
from pynamics.kinematics.topology import Topology

from pynamics.math.linalg import R3_cross_product_matrix

class JointDamping(JointDynamicsParent):
    def __init__(
            self,
            damping_factor: float,
            joint_names:typing.List[str] = []
        ):
        super().__init__(joint_names=joint_names)
        self.damping_factor = damping_factor
    
    def compute_dynamics(self, topology:Topology, body_name:str) -> typing.Tuple[np.matrix, np.matrix]:
        return [-self.damping_factor * topology.joints[body_name].get_configuration_d(),]
