"""
Module implementing a Dynamics source representing a constant force.
"""

import typing

import numpy as np

from pynamics.dynamics import BodyDynamicsParent
from pynamics.kinematics.topology import Topology


class ConstantBodyForce(BodyDynamicsParent):
    def __init__(
        self,
        force: np.matrix,
        application_position: typing.Tuple[str, str] = ("World", "Identity"),
        application_orientation: typing.Tuple[str, str] = ("World", "Identity"),
        body_names: typing.List[str] = [],
    ):
        """
        Apply a constant force to a body

        Args:
            force (np.matrix): The constant force to apply.
            application_position (typing.Tuple[str, str]): The (body, frame) \
            pair in the Topology representing the location at which to apply the \
            force. Defaults to (\"World\", \"Identity\") \
            application_orientation (typing.Tuple[str, str]): The (body, frame) \
            pair in the Topology representing the orientation in which to express \
            the force. Defaults to (\"World\", \"Identity\") \
            body_names (typing.List[str], optional): List of bodies on which to \
            apply the constant force. Defaults to [].
        """
        super().__init__(body_names=body_names)
        self.force = force
        self.application_orientation = application_orientation
        self.application_position = application_position

    def compute_dynamics(
        self, topology: Topology, body_name: str
    ) -> typing.Tuple[np.matrix, np.matrix]:

        T_position = topology.get_transform(
            from_body_name="World",
            from_frame_name="Identity",
            to_body_name=self.application_position[0], 
            to_frame_name=self.application_position[1],
        )

        T_orientation = topology.get_transform(
            from_body_name=self.application_orientation[0], 
            from_frame_name=self.application_orientation[1],
            to_body_name="World",
            to_frame_name="Identity"
        )

        return [
            (T_orientation[:3,:3]@self.force, T_position[:3,3]),
        ], {}
