"""
Module implementing a Dynamics source representing a constant force.
"""

import typing

import numpy as np

from pynamics.dynamics import BodyDynamicsParent
from pynamics.kinematics.topology import Topology


class ConstantForce(BodyDynamicsParent):
    def __init__(
        self,
        force: np.matrix,
        point_of_application: np.matrix = np.matrix([0,0,0]).T,
        body_names: typing.List[str] = [],
    ):
        """
        Apply a constant force to a body

        Args:
            force (np.matrix): The constant force to apply.
            point_of_application (np.matrix, optional): The position in the body \
            frame at which to apply the force. Defaults to np.matrix([0,0,0]).T.
            body_names (typing.List[str], optional): List of bodies on which to \
            apply the constant force. Defaults to [].
        """
        super().__init__(body_names=body_names)
        self.force = force

    def compute_dynamics(
        self, topology: Topology, body_name: str
    ) -> typing.Tuple[np.matrix, np.matrix]:
        force = self.g * self.direction * topology.bodies[body_name].mass

        return [
            (self.force, self.point_of_application),
        ]
