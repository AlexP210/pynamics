"""Base classes for Dynamics Modules"""

import typing
import abc
import numpy as np

from pynamics.kinematics.topology import Topology
from pynamics.math.linalg import R3_cross_product_matrix


class BodyDynamicsParent(abc.ABC):
    def __init__(self, body_names: typing.List[str] = []):
        """Initialize a BodyDynamics module.

        Args:
            body_names (typing.List[str], optional): A list of body names on \
            which to apply forces. Defaults to [].
        """
        self.body_names = body_names

    @abc.abstractmethod
    def compute_dynamics(
        self, topology: Topology, body_name: str
    ) -> typing.Tuple[
        typing.List[typing.Tuple[np.matrix, np.matrix]], typing.Dict[str, float]
    ]:
        """Compute a list of forces and their points of application in the \
        body frame.

        Args:
            topology (Topology): The `Topology` object on which to compute forces.
            body_name (str): The name of the body on which to compute forces.

        Raises:
            NotImplementedError: Default implementation.

        Returns:
            typing.List[typing.Tuple[np.matrix, np.matrix]]: A list of (force, position) \
            pairs, representing the applied forces and where they are applied in world co-ordinates.
            typing.Dict[str, float]: A dictionary with any associated data that should be logged
        """
        raise NotImplementedError(
            "Implement `compute_dynamics()` in your `BodyDynamicsParent` subclass."
        )

    def __call__(self, topology: Topology, body_name: str) -> np.matrix:
        total_wrench = np.matrix(np.zeros(shape=(6, 1)))
        data = {}
        if body_name in self.body_names or self.body_names == []:
            forces_and_points_of_application, data = self.compute_dynamics(topology, body_name)
            for force, point_of_application in forces_and_points_of_application:
                wrench = np.matrix(np.zeros(shape=(6, 1)))
                wrench[:3, 0] = R3_cross_product_matrix(point_of_application) @ force
                wrench[3:, 0] = force
                total_wrench += wrench
            data["Total Moment"] = np.linalg.norm(total_wrench[:3])
            data["Total Force"] = np.linalg.norm(total_wrench[3:])
            for idx in range(6):
                data[f"Force {idx}"] = total_wrench[idx,0]

        return total_wrench, data

    def update(self, topology: Topology, dt: float):
        """Perform any update steps needed by the dynamics module.

        Args:
            topology (Topology): `Topology` object containing state to use for update.
            dt (float): Time step by which to increment the state.
        """
        pass

    def get_data(self) -> typing.Dict[str, float]:
        """Data to add to the `Sim`'s `data_history` on each timestep.

        Returns:
            typing.Dict[str, float]: Dictionary mapping data key to data value.
        """
        return {}


class JointDynamicsParent(abc.ABC):
    def __init__(self, joint_names: typing.List[str] = []):
        """Initialize a JointDynamics module.

        Args:
            joint_names (typing.List[str], optional): The list of joint names on \
            which to apply forces. Defaults to [].
        """
        self.joint_names = joint_names

    @abc.abstractmethod
    def compute_dynamics(
        self, topology: Topology, joint_name: str
    ) -> typing.Tuple[typing.List[np.matrix], typing.Dict[str, float]]:
        """Compute a list of joint space forces.

        Args:
            topology (Topology): The `Topology` object on which to compute forces.
            joint_name (str): The name of the joint on which to compute forces.

        Raises:
            NotImplementedError: Default implementation.

        Returns:
            typing.List[np.matrix]: List of joint space forces being applied.
            typing.Dict[str, float]: A dictionary with any associated data that should be logged.
        """
        raise NotImplementedError(
            "Implement `compute_dynamics()` in your `JointDynamicsParent` subclass."
        )

    def __call__(self, topology: Topology, joint_name: str) -> np.matrix:
        if (joint_name in self.joint_names) or (self.joint_names == []):
            forces, data = self.compute_dynamics(topology, joint_name)
            total_force = sum(forces)
            data["Total Force"] = np.linalg.norm(total_force)
            for idx in range(total_force.size):
                data[f"Force {idx}"] = total_force[idx,0]
            return total_force, data
        else:
            return np.zeros(
                shape=topology.joints[joint_name].get_configuration_d().shape
            ), {}

    def update(self, topology: Topology, dt: float):
        """Perform any update steps needed by the dynamics module.

        Args:
            topology (Topology): `Topology` object containing state to use for update.
            dt (float): Time step by which to increment the state.
        """
        pass

    def get_data(self) -> typing.Dict[str, float]:
        """Data to add to the `Sim`'s `data_history` on each timestep.

        Returns:
            typing.Dict[str, float]: Dictionary mapping data key to data value.
        """
        return {}
