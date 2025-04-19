"""
This module contains the definition of `Sim`, the core Pynamics module
for running simulations.
"""

import typing

import numpy as np
import pandas as pd
import tqdm

from pynamics.dynamics import (
    BodyDynamicsParent,
    JointDynamicsParent,
)
from pynamics.math import linalg
from pynamics.math import integrators
from pynamics.kinematics.topology import Topology


class Sim:
    """
    The core class representing a Pynamics simulation
    """

    # pylint: disable=invalid-name
    # Many variables are named to match those used in Featherstone
    def __init__(
        self,
        topology: Topology,
        body_dynamics: typing.Dict[str, BodyDynamicsParent] = None,
        joint_dynamics: typing.Dict[str, JointDynamicsParent] = None,
        integrator: integrators.Integrator = integrators.RungeKutta4(),
    ) -> None:
        """
        Initialize a `Sim`-ulation.

        Args:
            topology (Topology): A Topology object representing the initial condition.
            body_dynamics (typing.List[DynamicsParent], optional): A list of \
            BodyDynamics modules to apply forces to the topology. Defaults to [].
            joint_dynamics (typing.List[DynamicsParent], optional): A list of \
            JointDynamics modules to apply forces to the topology. Defaults to [].
        """
        self.body_dynamics = body_dynamics if body_dynamics is not None else {}
        self.joint_dynamics = joint_dynamics if joint_dynamics is not None else {}
        self.topology = topology
        self.integrator = integrator
        self._initialize_data_dict()

    def _initialize_data_dict(self):
        self.data: typing.Dict = {}

        self.data["Bodies"] = {}
        self.data["Joints"] = {}
        for body_name in self.topology.get_ordered_body_list():
            self.data["Bodies"][f"{body_name}"] = {}
            for i in range(7):
                self.data["Bodies"][f"{body_name}"][f"Position {i}"] = []
            for i in range(6):
                self.data["Bodies"][f"{body_name}"][f"Velocity {i}"] = []
                self.data["Bodies"][f"{body_name}"][f"Acceleration {i}"] = []

            self.data["Joints"][f"{body_name}"] = {}
            joint = self.topology.joints[body_name]
            for i in range(joint.get_configuration().size):
                self.data["Joints"][f"{body_name}"][f"Position {i}"] = []
            for i in range(joint.get_configuration_d().size):
                self.data["Joints"][f"{body_name}"][f"Velocity {i}"] = []
                self.data["Joints"][f"{body_name}"][f"Acceleration {i}"] = []

        self.data["Body Forces"] = {}
        self.data["Joint Forces"] = {}
        for body_dynamics_name, body_dynamics_module in self.body_dynamics.items():
            self.data["Body Forces"][body_dynamics_name] = {}
            for body_name in self.topology.get_ordered_body_list():
                self.data["Body Forces"][body_dynamics_name][body_name] = {}
        for joint_dynamics_name, joint_dynamics_module in self.joint_dynamics.items():
            self.data["Joint Forces"][joint_dynamics_name] = {}
            for body_name in self.topology.get_ordered_body_list():
                self.data["Joint Forces"][joint_dynamics_name][body_name] = {}

    def inverse_dynamics(
        self,
        joint_space_positions: typing.Dict[str, np.matrix],
        joint_space_velocities: typing.Dict[str, np.matrix],
        joint_space_accelerations: typing.Dict[str, np.matrix],
    ) -> typing.Dict[str, np.matrix]:
        """Calculate the inverse dynamics for the topology.

        Args:
            joint_space_positions (typing.Dict[str, np.matrix]): The joint \
            space positions for configuring the topology.
            joint_space_velocities (typing.Dict[str, np.matrix]): The joint \
            space velocities for configuring the topology.
            joint_space_accelerations (typing.Dict[str, np.matrix]): The joint \
            space accelerations for which to calculate the applied joint space \
            forces.

        Returns:
            typing.Dict[str, np.matrix]: Dictionary mapping joint names to joint \
            space forces.
        """
        # Set the topology
        self.topology.set_joint_positions(joint_space_positions)
        self.topology.set_joint_velocities(joint_space_velocities)

        body_names = self.topology.get_ordered_body_list()
        body_accelerations = self.topology.calculate_body_accelerations(
            joint_space_accelerations
        )

        S = {}
        total_joint_forces = {}
        motion_subspace_forces = {}
        for body_name in body_names[1:]:
            joint = self.topology.joints[body_name]
            parent_body_name, parent_frame_name = self.topology.tree[body_name]
            S_i = joint.get_motion_subspace()
            S[body_name] = S_i
            i_X_0_star = self.topology.get_Xstar(
                "World", "Identity", body_name, "Identity"
            )  # linalg.X_star(i_X_0)
            f1 = (
                self.topology.bodies[body_name].mass_matrix
                @ body_accelerations[body_name]
            )
            f2 = (
                linalg.cross_star(self.topology.bodies[body_name].get_velocity())
                @ self.topology.bodies[body_name].mass_matrix
                @ self.topology.bodies[body_name].get_velocity()
            )
            f3 = np.matrix(np.zeros(shape=(6, 1)))
            for name, dynamics_module in self.body_dynamics.items():
                force, data = dynamics_module(self.topology, body_name)
                # Add to data
                for key, value in data.items():
                    if key in self.data["Body Forces"][name][body_name]:
                        self.data["Body Forces"][name][body_name][key].append(value)
                    else:
                        self.data["Body Forces"][name][body_name][key] = [
                            value,
                        ]
                force = i_X_0_star @ force
                f3 += force
            total_joint_forces[body_name] = f1 + f2 - f3
        for body_name in body_names[-1:0:-1]:
            motion_subspace_forces[body_name] = (
                S[body_name].T @ total_joint_forces[body_name]
            )
            parent_body_name, _ = self.topology.tree[body_name]
            if parent_body_name != "World":
                lambda_i__Xstar__i = self.topology.get_Xstar(
                    body_name, "Identity", parent_body_name, "Identity"
                )
                total_joint_forces[parent_body_name] += (
                    lambda_i__Xstar__i @ total_joint_forces[body_name]
                )
        return motion_subspace_forces

    def get_nonlinear_forces(
        self,
        joint_space_positions: typing.Dict[str, np.matrix],
        joint_space_velocities: typing.Dict[str, np.matrix],
    ) -> typing.Dict[str, np.matrix]:
        """Calculates the joint-space forces due to non-linear forces from \
        the velocities of each body.

        Args:
            joint_space_positions (typing.Dict[str, np.matrix]): The joint \
            space positions for configuring the topology.
            joint_space_velocities (typing.Dict[str, np.matrix]): The joint \
            space velocities for configuring the topology.
        
        Returns:
            typing.Dict[str, np.matrix]: Dictionary mapping joint name to non-\
            linear joint-space forces at that joint.
        """
        body_names = self.topology.get_ordered_body_list()[1:]
        # Zero acceleration on all joints
        joint_space_accelerations = {
            body_name: np.zeros(
                shape=(self.topology.joints[body_name].get_configuration_d().size, 1)
            )
            for body_name in body_names
        }

        # Get the ordered list of bodies to iterate over
        body_names = self.topology.get_ordered_body_list()

        # Get the force needed across each joint to maintian zero acceleration
        nonlinear_joint_space_forces = self.inverse_dynamics(
            joint_space_positions=joint_space_positions,
            joint_space_velocities=joint_space_velocities,
            joint_space_accelerations=joint_space_accelerations,
        )
        return nonlinear_joint_space_forces

    def forward_dynamics(
        self,
        joint_space_positions: typing.Dict[str, np.matrix],
        joint_space_velocities: typing.Dict[str, np.matrix],
        joint_space_forces: typing.Dict[str, np.matrix],
    ) -> typing.Dict[str, np.matrix]:
        """Calculate the forward dynamics for the topology, i.e. joint space \
        accelerations given joint-space forces.

        Args:
            joint_space_forces (typing.Dict[str, np.matrix]): Dictionary mapping \
            joint name to the applied joint space force.

        Returns:
            typing.Dict[str, np.matrix]: Dictionary mapping joint name to joint \
            space acceleration.
        """
        # Get the nonlinear joint-space force vector
        C = self.topology.vectorify_velocity(self.get_nonlinear_forces(
            joint_space_positions=joint_space_positions,
            joint_space_velocities=joint_space_velocities,
        ))

        # Set the topology & get the joint-space mass matrix
        self.topology.set_joint_positions(joint_space_positions)
        self.topology.set_joint_velocities(joint_space_velocities)
        H = self.topology.matrixify(self.topology.get_mass_matrix())

        # Calculate the joint space accelerations, return in dict format
        tau = self.topology.vectorify_velocity(joint_space_forces)
        q_dd = np.linalg.inv(H) @ (tau - C)
        joint_space_accelerations = self.topology.dictionarify_velocity(q_dd)
        return joint_space_accelerations

    def _derivative_function(self, Q):

        body_names = self.topology.get_ordered_body_list()[1:]
        n_position_dof = sum([self.topology.joints[name].get_configuration().size for name in body_names])
        joint_space_positions = self.topology.dictionarify_position(Q[:n_position_dof])
        joint_space_velocities = self.topology.dictionarify_velocity(Q[n_position_dof:])

        # Get the q_d part of Q_d
        P = self.topology.matrixify(self.topology._get_P())
        zeros = np.zeros(shape=(n_position_dof, n_position_dof))
        A = np.concatenate([zeros, P], axis=1)
        q_d = A @ Q

        # Calculate the joint space forces
        joint_space_forces = self.calculate_joint_space_forces(
            joint_space_positions=joint_space_positions,
            joint_space_velocities=joint_space_velocities,
        )

        # Get the q_dd part of Q_d
        q_dd = self.topology.vectorify_velocity(self.forward_dynamics(
            joint_space_positions=joint_space_positions,
            joint_space_velocities=joint_space_velocities,
            joint_space_forces=joint_space_forces
        ))
        return np.concatenate([q_d, q_dd])

    def step(self, dt: float) -> None:
        """Advance the simulation by one step.

        Args:
            dt (float): The time step by which to advance the simulation.
        """
        # Log the current time
        if "Time" not in self.data:
            self.data["Time"] = [
                0,
            ]
        else:
            self.data["Time"].append(self.data["Time"][-1] + dt)

        # Log the joint positions & velocities
        joint_space_positions = self.topology.get_joint_space_positions()
        self._log_joint_positions(joint_space_positions)
        joint_space_velocities = self.topology.get_joint_space_velocities()
        self._log_joint_velocities(joint_space_velocities)

        # Log the body positions & velocities
        body_positions = self.topology.get_body_positions()
        self._log_body_positions(body_positions)
        body_velocities = self.topology.get_body_velocities()
        self._log_body_velocities(body_velocities)

        # Run the 
        Q, Q_d = self.integrator.integrate(
            dt=dt,
            derivative_function=self._derivative_function
        )

        body_names = self.topology.get_ordered_body_list()[1:]
        n_position_dof = sum([self.topology.joints[name].get_configuration().size for name in body_names])
        joint_space_positions = self.topology.dictionarify_position(Q[:n_position_dof])
        joint_space_velocities = self.topology.dictionarify_velocity(Q[n_position_dof:])
        joint_space_accelerations = self.topology.dictionarify_velocity(Q_d[n_position_dof:])

        # Calculate & log the body accelerations
        cartesian_space_accelerations = self.topology.calculate_body_accelerations(
            joint_space_accelerations
        )
        for body_name, body in self.topology.bodies.items():
            T = self.topology.get_X(
                from_body_name=body_name,
                from_frame_name="Identity",
                to_body_name="World",
                to_frame_name="Identity",
            )
            transformed_acceleration = T @ cartesian_space_accelerations[body_name]
            for dof_idx in range(6):
                self.data["Bodies"][body_name][f"Acceleration {dof_idx}"].append(
                    transformed_acceleration[dof_idx, 0]
                )

        # Update the topology with the integrated state
        self.topology.set_joint_positions(joint_space_positions)
        self.topology.set_joint_velocities(joint_space_velocities)

        # Update the dynamics modules
        for name, dynamics_module in self.body_dynamics.items():
            dynamics_module.update(self.topology, dt)
        for name, dynamics_module in self.joint_dynamics.items():
            dynamics_module.update(self.topology, dt)

    def calculate_joint_space_forces(
        self,
        joint_space_positions: typing.Dict[str, np.matrix],
        joint_space_velocities: typing.Dict[str, np.matrix],
    ) -> typing.Dict[str, np.matrix]:
        """
        Calculate the joint-space forces due to the joint-space force modules.

        Args:
            joint_space_positions (typing.Dict[str, np.matrix]): Joint space \
            positions for configuring the topology.
            joint_space_velocities (typing.Dict[str, np.matrix]): Joint space \
            velocities for configuring the topology.

        Returns:
            typing.Dict[str, np.matrix]: Dictionary mapping joint name to the \
            joint space force acting across it.
        """
        self.topology.set_joint_positions(joint_space_positions)
        self.topology.set_joint_velocities(joint_space_velocities)

        joint_space_forces = {}
        body_names = self.topology.get_ordered_body_list()
        for body_name in body_names[1:]:
            q_shape = self.topology.joints[body_name].get_configuration_d().shape
            joint_space_forces[body_name] = np.matrix(np.zeros(shape=q_shape))
            for dynamics_module_name, dynamics_module in self.joint_dynamics.items():
                forces, data = dynamics_module(self.topology, body_name)
                for key, value in data.items():
                    if (
                        key
                        in self.data["Joint Forces"][dynamics_module_name][body_name]
                    ):
                        self.data["Joint Forces"][dynamics_module_name][body_name][
                            key
                        ].append(value)
                    else:
                        self.data["Joint Forces"][dynamics_module_name][body_name][
                            key
                        ] = [
                            value,
                        ]
                joint_space_forces[body_name] += forces
        return joint_space_forces

    def _log_joint_positions(self, joint_space_positions):
        for joint_name, joint in self.topology.joints.items():
            for dof_idx in range(joint.get_configuration().size):
                self.data["Joints"][joint_name][f"Position {dof_idx}"].append(
                    joint_space_positions[joint_name][dof_idx, 0]
                )

    def _log_joint_velocities(self, joint_space_velocities):
        for joint_name, joint in self.topology.joints.items():
            for dof_idx in range(joint.get_configuration_d().size):
                self.data["Joints"][joint_name][f"Velocity {dof_idx}"].append(
                    joint_space_velocities[joint_name][dof_idx, 0]
                )

    def _log_joint_accelerations(self, joint_space_accelerations):
        for joint_name, joint in self.topology.joints.items():
            for dof_idx in range(joint.get_configuration_d().size):
                self.data["Joints"][joint_name][f"Acceleration {dof_idx}"].append(
                    joint_space_accelerations[joint_name][dof_idx, 0]
                )

    def _log_body_positions(self, body_positions):
        for body_name in self.topology.get_ordered_body_list()[1:]:
            for dof in range(7):
                self.data["Bodies"][body_name][f"Position {dof}"].append(
                    body_positions[body_name][dof, 0]
                )

    def _log_body_velocities(self, body_velocities):
        for body_name in self.topology.get_ordered_body_list()[1:]:
            for dof in range(6):
                self.data["Bodies"][body_name][f"Velocity {dof}"].append(
                    body_velocities[body_name][dof, 0]
                )

    def simulate(self, delta_t: float, dt: float, verbose=False) -> None:
        """Advance the simulation.

        Args:
            delta_t (float): The length of time to simulate.
            dt (float): The timestep to advance the simulation by in each step.
            verbose (bool, optional): Whether to display progress bar. Defaults to False.
        """
        # Make sure velocities are updated
        # If they're not, then body velocities will not be set;
        # Any dynamics that uses body velocity will calculate with stale
        # values on the first step
        self.topology.update_body_velocities()

        # Initialize the integrators with the current state
        self.integrator.set_initial_condition(
            initial_condition=self.topology._get_state_vector()
        )

        if verbose:
            for _ in tqdm.tqdm(range(int(delta_t // dt + 1))):
                self.step(dt=dt)
        else:
            for _ in range(int(delta_t // dt + 1)):
                self.step(dt=dt)

    def save_data(self, path: str):
        """Save the collected sim data to CSV.

        Args:
            path (str): Path at which to save the collected data.
        """
        pd.DataFrame(self.data).to_csv(path, index=False, sep=",")

    def _add_to_data(self, data_field_name, data_value):
        if data_field_name in self.data:
            self.data[data_field_name].append(data_value)
        else:
            self.data[data_field_name] = [
                data_value,
            ]
