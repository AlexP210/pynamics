"""
Module implementing a Dynamics source representing joint space viscuous friction.
"""

import typing

import numpy as np

from pynamics.dynamics import JointDynamicsParent
from pynamics.kinematics.topology import Topology


class ConstantJointForce(JointDynamicsParent):

    def __init__(self, force: np.matrix, joint_names: typing.List[str] = []):
        """
        Apply a constant joint space force.

        Args:
            force (np.matrix): Joint space force to apply.
            joint_names (typing.List[str], optional): List of joints to apply the force to. Defaults to [].
        """
        super().__init__(joint_names=joint_names)
        self.force = force

    def compute_dynamics(
        self, topology: Topology, body_name: str
    ) -> typing.List[np.matrix]:
        return [
            self.force,
        ], {}
