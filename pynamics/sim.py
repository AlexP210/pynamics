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
        self.body_dynamics: typing.List[BodyDynamicsParent] = (
            body_dynamics if body_dynamics is not None else {}
        )
        self.joint_dynamics: typing.List[JointDynamicsParent] = (
            joint_dynamics if joint_dynamics is not None else {}
        )
        self.topology: Topology = topology
        self.joint_space_position_history = []
        self.joint_space_velocity_history = []
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
        self, joint_space_accelerations: typing.Dict[str, np.matrix]
    ) -> typing.Dict[str, np.matrix]:
        """Calculate the inverse dynamics for the topology.

        Args:
            joint_space_accelerations (typing.Dict[str, np.matrix]): The joint \
            space accelerations for which to calculate the applied joint space \
            forces.

        Returns:
            typing.Dict[str, np.matrix]: Dictionary mapping joint names to joint \
            space forces.
        """
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
                        self.data["Body Forces"][name][body_name][key] = [value,]
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

    def get_nonlinear_forces(self) -> typing.Dict[str, np.matrix]:
        """Calculates the joint-space forces due to non-linear forces from \
        the velocities of each body.

        Returns:
            typing.Dict[str, np.matrix]: Dictionary mapping joint name to non-\
            linear joint-space forces at that joint.
        """
        # Get the ordered list of bodies to iterate over
        body_names = self.topology.get_ordered_body_list()
        # Initialize the states to track velocity & acceleration
        nonlinear_joint_space_forces = self.inverse_dynamics(
            joint_space_accelerations={
                body_name: np.matrix(
                    np.zeros(
                        (self.topology.joints[body_name].get_configuration_d().size)
                    )
                ).T
                for body_name in body_names
            }
        )
        return nonlinear_joint_space_forces

    def forward_dynamics(
        self, joint_space_forces: typing.Dict[str, np.matrix]
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
        C = self.topology.vectorify_velocity(self.get_nonlinear_forces())
        H = self.topology.matrixify(self.topology.get_mass_matrix())
        tau = self.topology.vectorify_velocity(joint_space_forces)
        q_dd = np.linalg.inv(H) @ (tau - C)
        joint_space_accelerations = self.topology.dictionarify(q_dd)
        return joint_space_accelerations

    def step(self, dt: float) -> None:
        """Advance the simulation by one step.

        Args:
            dt (float): The time step by which to advance the simulation.
        """
        # Save the current state
        if "Time" not in self.data:
            self.data["Time"] = [
                0,
            ]
        else:
            self.data["Time"].append(self.data["Time"][-1] + dt)
        joint_space_positions = self.topology.get_joint_space_positions()
        joint_space_velocities = self.topology.get_joint_space_velocities()
        for joint_name, joint in self.topology.joints.items():
            for dof_idx in range(joint.get_configuration().size):
                self.data["Joints"][joint_name][f"Position {dof_idx}"].append(
                    joint_space_positions[joint_name][dof_idx,0]
                )
            for dof_idx in range(joint.get_configuration_d().size):
                self.data["Joints"][joint_name][f"Velocity {dof_idx}"].append(
                    joint_space_velocities[joint_name][dof_idx,0]
                )
        for body_name, body in self.topology.bodies.items():
            T = self.topology.get_transform(
                from_body_name="World", from_frame_name="Identity",
                to_body_name=body_name, to_frame_name="Identity"
            )
            r = T[:3,3]
            C = T[:3,:3]
            eig_val, eig_vec = np.linalg.eig(C)
            index = (abs(eig_val.real - 1) < 0.0000001) & (abs(eig_val.imag) <  0.0000001)
            axis = eig_vec[:,index][:,0]
            cos_angle = (np.linalg.trace(C) - 1) / 2
            angle = np.arccos(cos_angle)
            q = np.matrix([
                float(np.cos(angle/2)), 
                float(np.sin(angle/2)*axis[0,0].real), 
                float(np.sin(angle/2)*axis[1,0].real),
                float(np.sin(angle/2)*axis[2,0].real)
            ]).T
            for dof in range(4):
                self.data["Bodies"][body_name][f"Position {dof}"].append(q[dof,0])
            for dof in range(3):
                self.data["Bodies"][body_name][f"Position {dof}"].append(r[dof,0])
            T = self.topology.get_X(
                from_body_name=body_name, from_frame_name="Identity",
                to_body_name="World", to_frame_name="Identity"
            )
            transformed_velocity = T@body.get_velocity()
            for dof in range (6):
                self.data["Bodies"][body_name][f"Velocity {dof}"].append(
                    transformed_velocity[dof,0]
                )

        # TODO: Update Visualizer to no longer need these so they can be deprecated
        self.joint_space_position_history.append(
            self.topology.get_joint_space_positions()
        )
        self.joint_space_velocity_history.append(
            self.topology.get_joint_space_velocities()
        )

        joint_space_forces = {}
        body_names = self.topology.get_ordered_body_list()
        for body_name in body_names[1:]:
            q_shape = self.topology.joints[body_name].get_configuration_d().shape
            joint_space_forces[body_name] = np.matrix(np.zeros(shape=q_shape))
            for dynamics_module_name, dynamics_module in self.joint_dynamics.items():
                forces, data = dynamics_module(
                    self.topology, body_name
                )
                for key, value in data.items():
                    if key in self.data["Joint Forces"][dynamics_module_name][body_name]:
                        self.data["Joint Forces"][dynamics_module_name][body_name][key].append(value)
                    else:
                        self.data["Joint Forces"][dynamics_module_name][body_name][key] = [value,]
                joint_space_forces[body_name] += forces

            
        joint_space_accelerations = self.forward_dynamics(joint_space_forces)
        for joint_name, joint in self.topology.joints.items():
            for dof_idx in range(joint.get_configuration_d().size):
                self.data["Joints"][joint_name][f"Acceleration {dof_idx}"].append(
                    joint_space_accelerations[joint_name]
                )
        cartesian_space_accelerations = self.topology.calculate_body_accelerations(joint_space_accelerations)
        for body_name, body in self.topology.bodies.items():
            T = self.topology.get_X(
                from_body_name=body_name, from_frame_name="Identity",
                to_body_name="World", to_frame_name="Identity"
            )
            transformed_acceleration = T@cartesian_space_accelerations[body_name]
            for dof_idx in range(6):
                self.data["Bodies"][body_name][f"Acceleration {dof}"].append(
                    transformed_acceleration[dof,0]
                )


        for body_name in body_names[1:]:
            A = joint_space_accelerations[body_name]
            self.topology.joints[body_name].integrate(dt, A)
        # Update the velocities of each body based on the new velocities of each joint
        self.topology.update_body_velocities()

        # Update the dynamics modules
        for name, dynamics_module in self.body_dynamics.items():
            dynamics_module.update(self.topology, dt)
        for name, dynamics_module in self.joint_dynamics.items():
            dynamics_module.update(self.topology, dt)

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
        # Initialize the integrators with the current joint velocities
        for joint_name, joint in self.topology.joints.items():
            joint.initialize_integrator()

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
