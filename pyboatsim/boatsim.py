import typing
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm as tqdm

from pyboatsim.constants import HOME, AXES
from pyboatsim.state import State
from pyboatsim.dynamics import DynamicsParent, WaterWheel, SimpleBodyDrag, ConstantForce, MeshBuoyancy, Gravity, MeshBodyDrag
from pyboatsim.math import linalg
from pyboatsim.kinematics.topology import Topology, Frame, Body

class Sim:
    def __init__(
            self,
            topology: Topology,
            body_dynamics: typing.List[DynamicsParent] = [],
            joint_dynamics: typing.List[DynamicsParent] = [],
        ) -> None:
        """
        Initializer
        """
        self.body_dynamics:typing.List[DynamicsParent] = body_dynamics
        self.joint_dynamics:typing.List[DynamicsParent] = joint_dynamics
        self.topology:Topology = topology
        self.joint_space_position_history = [
            self.topology.get_joint_space_positions(),
        ]
        self.joint_space_velocity_history = [
            self.topology.get_joint_space_velocities(),
        ]
        self.time_history = [
            0
        ]

    def inverse_dynamics(self, q_dd):

        body_names = self.topology.get_ordered_body_list()

        body_accelerations = self.topology.calculate_body_accelerations(q_dd)
       
        S = {}
        total_joint_forces = {}
        motion_subspace_forces = {}
        for body_name in body_names[1:]:
            joint = self.topology.joints[body_name]
            parent_body_name, parent_frame_name = self.topology.tree[body_name]
            S_i = joint.get_motion_subspace()
            S[body_name] = S_i
            i_X_0 = self.topology.get_X("World", "Identity", body_name, "Identity")
            i_X_0_star = self.topology.get_Xstar("World", "Identity", body_name, "Identity") #linalg.X_star(i_X_0)

            f1 = self.topology.bodies[body_name].mass_matrix @ body_accelerations[body_name]
            f2 = linalg.cross_star(self.topology.bodies[body_name].get_velocity()) @ self.topology.bodies[body_name].mass_matrix @ self.topology.bodies[body_name].get_velocity()
            f3 = np.matrix(np.zeros(shape=(6,1)))
            for dynamics_module in self.body_dynamics:
                force = i_X_0_star @ dynamics_module(self.topology, body_name)
                f3 += force
            total_joint_forces[body_name] = f1+f2-f3
        for body_name in body_names[-1:0:-1]:
            motion_subspace_forces[body_name] = S[body_name].T @ total_joint_forces[body_name]
            parent_body_name, _ = self.topology.tree[body_name]
            if parent_body_name != "World":
                lambda_i__Xstar__i = self.topology.get_Xstar(body_name,"Identity",parent_body_name,"Identity")
                total_joint_forces[parent_body_name] += lambda_i__Xstar__i @ total_joint_forces[body_name]
        return motion_subspace_forces

    def get_nonlinear_forces(self):
        # Get the ordered list of bodies to iterate over
        body_names = self.topology.get_ordered_body_list()
        # Initialize the states to track velocity & acceleration
        tau = self.inverse_dynamics(
            q_dd={
                body_name: np.matrix(np.zeros((self.topology.joints[body_name].get_number_degrees_of_freedom())))
                for body_name in body_names
            }
        )
        return tau
    
    def forward_dynamics(self, joint_space_forces):
        C = self.topology.vectorify(self.get_nonlinear_forces())
        H = self.topology.get_mass_matrix()
        if type(joint_space_forces) == type(np.matrix(0)):
            tau = joint_space_forces
        elif type(joint_space_forces) == type(dict()):
            tau = self.topology.vectorify(joint_space_forces)
        q_dd = np.linalg.inv(H) @ (tau - C)
        body_accelerations = self.topology.calculate_body_accelerations(q_dd)
        joint_space_accelerations = self.topology.dictionarify(q_dd)
        c = self.topology.dictionarify(C)
        t = self.topology.dictionarify(tau)
        return joint_space_accelerations       

    def step(self, dt):
        """
        Steps the simulation by `self._state["dt"]`.
        """
        # Apply the dynamics on the state, adds forces, torques, and other
        # intermediate values calculated by dynamics modules based on the
        # current state.
        joint_space_forces = {}
        body_names = self.topology.get_ordered_body_list()
        for body_name in body_names[1:]:
            for dynamics_module in self.joint_dynamics:
                joint_space_forces[body_name] = dynamics_module(self.topology, body_name)
        joint_space_accelerations = self.forward_dynamics(joint_space_forces)
        for body_name in body_names[1:]:
            # Update the positions
            # x_1 = x_0 + x'_0 * dt + 0.5 * x''_1 * dt^2
            # x'_1 = x'_0 + x''_0 * dt
            # x_i+1 = 2 * x_i - x_i-1 + x''_i * dt^2
            # x'_i+1 = (x_i+1 - x_i-1) / (2*dt)
            # print(body_name)
            # print(joint_space_accelerations)
            self.topology.joints[body_name].set_configuration_d(
                self.topology.joints[body_name].get_configuration_d()
                + joint_space_accelerations[body_name] * dt
            )
            self.topology.joints[body_name].set_configuration(
                self.topology.joints[body_name].get_configuration()
                + self.topology.joints[body_name].get_configuration_d() * dt
            )

            # if len(self.joint_space_position_history) == 1:
            #     # Update joint position
            #     self.topology.joints[body_name].set_configuration(
            #         self.topology.joints[body_name].get_configuration()
            #         + self.topology.joints[body_name].get_configuration_d() * dt
            #         + 0.5 * joint_space_accelerations[body_name] * dt**2
            #     )
            #     # Update joint velocity
            #     self.topology.joints[body_name].set_configuration_d(
            #         self.topology.joints[body_name].get_configuration_d()
            #         + joint_space_accelerations[body_name] * dt
            #     )
            # else:
            #     # Update joint position
            #     self.topology.joints[body_name].set_configuration(
            #         2*self.topology.joints[body_name].get_configuration()
            #         - self.joint_space_position_history[-1][body_name]
            #         + joint_space_accelerations[body_name] * dt**2
            #     )
            #     # Update joint velocity
            #     self.topology.joints[body_name].set_configuration_d(
            #         (self.topology.joints[body_name].get_configuration()
            #         - self.joint_space_position_history[-1][body_name]) / (2*dt)
            #     )
        self.joint_space_position_history.append(self.topology.get_joint_space_positions())
        self.joint_space_velocity_history.append(self.topology.get_joint_space_velocities())
        self.time_history.append(self.time_history[-1]+dt)
        self.topology.update_body_velocities()

        return

    def simulate(self, delta_t:float, dt:float, verbose=False):
        """
        Runs the simulation for delta_t more seconds.
        """
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
    from pyboatsim.example.example_topology import get_pendulum
    np.seterr(all='raise')


    N = 1
    pendulum, pendulum_vis = get_pendulum(N)
    pendulum.joints["Arm 0"].set_configuration(np.matrix([0.8*np.pi/2]))
    sim = Sim(
        topology=pendulum, 
        body_dynamics=[
            Gravity("gravity", -9.81, 2)
        ])
    # pendulum_vis.view()
    # sim.step(0.01)

    sim.simulate(10, 0.01, verbose=True)
    # pendulum_vis.add_sim_data(sim)
    # pendulum_vis.animate(0.01, save_path=f"Test_Multibody_{N}.mp4")

    Y = []
    X = []
    for joint_space_positions in sim.joint_space_position_history:
        X.append(joint_space_positions["Arm 0"][0,0])

    plt.scatter(sim.time_history, X, c="r", marker="x", s=0.5)
    plt.plot(sim.time_history, [-0.2*np.pi/2 * np.cos(np.sqrt(9.81) * t) + np.pi/2 for t in sim.time_history])
    plt.show()
