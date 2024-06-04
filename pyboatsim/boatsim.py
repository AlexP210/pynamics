import typing
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm as tqdm

from pyboatsim.constants import HOME, AXES
from pyboatsim.state import State
from pyboatsim.dynamics import DynamicsParent, WaterWheel, SimpleBodyDrag, ConstantForce, MeshBuoyancy, MeshGravity, MeshBodyDrag
from pyboatsim.math import linalg
from pyboatsim.kinematics.topology import Topology, Frame, Body

class BoAtSim:
    def __init__(
            self,
            body_dynamics: typing.List[DynamicsParent],
            joint_dynamics: typing.List[DynamicsParent],
            topology: Topology
        ) -> None:
        """
        Initializer
        """
        self.history = []
        self.body_dynamics:typing.List[DynamicsParent] = body_dynamics
        self.joint_dynamics:typing.List[DynamicsParent] = joint_dynamics
        self.topology:Topology = topology

        self.generalized_force = State(self.topology)
        self.external_force = State(self.topology)

    def inverse_dynamics(self):
        # Get the ordered list of bodies to iterate over
        body_names = self.topology.get_ordered_body_list()
        # Initialize the states to track velocity & acceleration
        total_joint_forces = State(self.topology)
        self.generalized_force.clear()
        self.external_force.clear()
        self.topology.bodies[body_names[0]].set_velocity(np.matrix(np.zeros(6)).T)
        self.topology.bodies[body_names[0]].set_acceleration(np.matrix(np.zeros(6)).T)
        # Algorithm
        S = {}
        for body_name in body_names[1:]:
            joint = self.topology.joints[body_name]
            parent_body_name, parent_frame_name = self.topology.tree[body_name]
            X_J = joint.get_X()
            X_T = self.topology.get_X(parent_body_name, "Identity", parent_body_name, parent_frame_name)
            S_i = joint.get_motion_subspace()
            S[body_name] = S_i
            v_J = joint.get_velocity()
            c_J = joint.get_c()
            i__X__lambda_i = X_J @ X_T
            i_X_0 = self.topology.get_X("World", "Identity", body_name, "Identity")
            i_X_0_star = linalg.X_star(i_X_0)
            self.topology.bodies[body_name].set_velocity(i__X__lambda_i @ self.topology.bodies[parent_body_name].get_velocity() + v_J)
            
            a1 = i__X__lambda_i @ self.topology.bodies[parent_body_name].get_acceleration()
            a2 = joint.get_acceleration()
            a3 = c_J + linalg.cross(self.topology.bodies[body_name].get_velocity()) @ v_J
            self.topology.bodies[body_name].set_acceleration(a1 + a2 + a3)

            f1 = self.topology.bodies[body_name].mass_matrix @ self.topology.bodies[body_name].get_acceleration()
            f2 = linalg.cross_star(self.topology.bodies[body_name].get_velocity()) @ self.topology.bodies[body_name].mass_matrix @ self.topology.bodies[body_name].get_velocity()
            f3 = 0
            for dynamics_module in self.body_dynamics:
                force = i_X_0_star @ dynamics_module(self.topology, body_name)
                f3 += force
                self.external_force[body_name] += force

            for dynamics_module in self.joint_dynamics:
                force += S[body_name] @ dynamics_module(self.topology, body_name)
                f3 += force
                self.external_force[body_name] += force

            total_joint_forces[body_name] = f1+f2-f3

        for body_name in body_names[-1:0:-1]:
            self.generalized_force[body_name] = S[body_name].T @ total_joint_forces[body_name]
            parent_name, _ = self.topology.tree[body_name]
            if parent_name != "World":
                total_joint_forces[parent_name] += self.topology.get_Xstar(body_name,"Identity",parent_body_name,"Identity") @ total_joint_forces[body_name]

    def get_nonlinear_forces(self):
        # Get the ordered list of bodies to iterate over
        body_names = self.topology.get_ordered_body_list()
        # Initialize the states to track velocity & acceleration
        total_joint_forces = State(self.topology)
        generalized_force = State(self.topology)
        self.external_force.clear()
        self.topology.bodies[body_names[0]].set_velocity(np.matrix(np.zeros(6)).T)
        self.topology.bodies[body_names[0]].set_acceleration(np.matrix(np.zeros(6)).T)
        # Algorithm
        S = {}
        for body_name in body_names[1:]:
            joint = self.topology.joints[body_name]
            parent_body_name, parent_frame_name = self.topology.tree[body_name]
            X_J = joint.get_X()
            X_T = self.topology.get_X(parent_body_name, "Identity", parent_body_name, parent_frame_name)
            S_i = joint.get_motion_subspace()
            S[body_name] = S_i
            v_J = joint.get_velocity()
            c_J = joint.get_c()
            i__X__lambda_i = X_J @ X_T
            i_X_0 = self.topology.get_X("World", "Identity", body_name, "Identity")
            i_X_0_star = linalg.X_star(i_X_0)
            self.topology.bodies[body_name].set_velocity(i__X__lambda_i @ self.topology.bodies[parent_body_name].get_velocity() + v_J)
            
            a1 = i__X__lambda_i @ self.topology.bodies[parent_body_name].get_acceleration()
            a2 = joint.get_acceleration()
            a3 = c_J + linalg.cross(self.topology.bodies[body_name].get_velocity()) @ v_J
            self.topology.bodies[body_name].set_acceleration(a1 + a2 + a3)

            f1 = self.topology.bodies[body_name].mass_matrix @ self.topology.bodies[body_name].get_acceleration()
            f2 = linalg.cross_star(self.topology.bodies[body_name].get_velocity()) @ self.topology.bodies[body_name].mass_matrix @ self.topology.bodies[body_name].get_velocity()
            f3 = 0
            for dynamics_module in self.body_dynamics:
                force = i_X_0_star @ dynamics_module(self.topology, body_name)
                f3 += force
                self.external_force[body_name] += force

            for dynamics_module in self.joint_dynamics:
                force += S[body_name] @ dynamics_module(self.topology, body_name)
                f3 += force
                self.external_force[body_name] += force

            total_joint_forces[body_name] = f1+f2-f3

        for body_name in body_names[-1:0:-1]:
            self.generalized_force[body_name] = S[body_name].T @ total_joint_forces[body_name]
            parent_name, _ = self.topology.tree[body_name]
            if parent_name != "World":
                total_joint_forces[parent_name] += self.topology.get_Xstar(body_name,"Identity",parent_body_name,"Identity") @ total_joint_forces[body_name]

        return generalized_force

    def forward_dynamics(self):
        C = self.get_nonlinear_forces()
        H = self.topology.get_mass_matrix()
        number_of_degrees_of_freedom = sum([j.get_number_of_degrees_of_freedom() for j in self.joints])
        s = 0
        tau = np.matrix(np.zeroes((number_of_degrees_of_freedom,1)))
        for body_name in self.topology.get_ordered_body_list():
            n_dof = self.topology.joints[body_name].get_number_of_degrees_of_freedom()
            tau[s:s+n_dof] = self.topology.joints[body_name].get_generalized_force()
            s+=n_dof
        q_dd = np.linalg.inv(H) @ (tau - C)
        s = 0
        for body_name in self.topology.get_ordered_body_list():
            self.topology.joints[body_name].set_configuration_dd(tau[s:s+n_dof])
        

    def step(self, dt):
        """
        Steps the simulation by `self._state["dt"]`.
        """
        # Apply the dynamics on the state, adds forces, torques, and other
        # intermediate values calculated by dynamics modules based on the
        # current state.
        for dynamics_module in self.dynamics:
            self.state = dynamics_module(self.state, self.topology, dt)
        for axis in AXES:
            # Calculate the total force & moment by adding all the "f_"  and
            # "tau_" labels in the state dictionary
            self.state.set(
                partial_state_dictionary={
                    f"f_{axis}__total": sum([
                        self.state[f"f_{axis}__{name}"] for name in self.dynamics_names
                    ]),
                    f"tau_{axis}__total": sum([
                        self.state[f"tau_{axis}__{name}"] for name in self.dynamics_names
                    ])
                }
            )

        # Solve Newton-Euler equations to calculate the linear and angular accelerations
        # in the state dictionary
        self._compute_accelerations()
        
        # Add the state to the history
        self.history.append(self.state.get())
        # Create a new state to store the next update
        next_state = self.state.copy()

        # For each axis, update the position and velocities to be used in the
        # next state
        if len(self.history) < 1: raise ValueError("Why is self.history empty? For shame.")
        for axis in AXES:
            if len(self.history) == 1:
                # Update the positions
                next_state[f"r_{axis}__boat"] += self.state[f"v_{axis}__boat"]*dt + 0.5*next_state[f"a_{axis}__boat"]*dt**2
                next_state[f"theta_{axis}__boat"] += self.state[f"omega_{axis}__boat"]*dt + 0.5*next_state[f"alpha_{axis}__boat"]*dt**2
                next_state[f"v_{axis}__boat"] += self.state[f"a_{axis}__boat"] * dt
                next_state[f"omega_{axis}__boat"] += self.state[f"alpha_{axis}__boat"] * dt
            else:
                # Update the positions
                x_prev = self.history[-2]
                lin = f"r_{axis}__boat"
                ang = f"theta_{axis}__boat"
                next_state[lin] = 2*self.state[lin] - x_prev[lin] + self.state[f"a_{axis}__boat"]*dt**2
                next_state[ang] = 2*self.state[ang] - x_prev[ang] + self.state[f"alpha_{axis}__boat"]*dt**2
                next_state[f"v_{axis}__boat"] = (next_state[lin] - x_prev[lin]) / (2*dt)
                next_state[f"omega_{axis}__boat"] = (next_state[ang] - x_prev[ang]) / (2*dt)


        # Set the next state
        self.state = next_state
        self.state["t"] += dt

    def simulate(self, delta_t:float, dt:float, verbose=False):
        """
        Runs the simulation for delta_t more seconds.
        """
        # Ensure that the state contains the basic required labels to run
        # a simulation
        missing_labels = [
            label 
            for label in self.required_labels if not label in self.state._state_dictionary
        ]
        if len(missing_labels) != 0:
            raise ValueError(
                f"Cannot compute dynamics, missing the following"
                f" labels: {', '.join(missing_labels)}"
            )
        if verbose:
            for _ in tqdm.tqdm(range(int(delta_t//dt+1))):
                self.step(dt=dt)
        else:
            for _ in range(int(delta_t//dt+1)):
                self.step(dt=dt)
    
    def save_history(self, file_path:str):
        self.get_history_as_dataframe().to_csv(file_path)

    def get_history_as_dataframe(self):
        return pd.DataFrame.from_dict(self.history)

if __name__ == "__main__":
    from pyboatsim.example.example_topology import robot, vis

    # robot.joints["Pitch Body 1"].set_configuration(np.matrix(np.pi/4))
    # robot.joints["Pitch Body 2"].set_configuration(np.matrix(-np.pi/4))

    robot.joints["Pitch Body 2"].set_configuration_dd((np.matrix(np.pi/4))/100)

    sim = BoAtSim(topology=robot, dynamics=[])

    for body_name, force in sim.inverse_dynamics().data.items():
        if type(force) != type([]): print(body_name, force.T)

    # vis.view()
    


