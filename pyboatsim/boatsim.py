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
        """Initialize a `Sim`ulation.

        Args:
            topology (Topology): A Topology object representing the initial condition.
            body_dynamics (typing.List[DynamicsParent], optional): A list of BodyDynamics modules to apply forces to the topology. Defaults to [].
            joint_dynamics (typing.List[DynamicsParent], optional): A list of JointDynamics modules to apply forces to the topology. Defaults to [].
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

    def inverse_dynamics(self, joint_space_accelerations:typing.Dict[str, np.matrix]) -> typing.Dict[str, np.matrix]:
        body_names = self.topology.get_ordered_body_list()

        body_accelerations = self.topology.calculate_body_accelerations(joint_space_accelerations)
       
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

    def get_nonlinear_forces(self) -> typing.Dict[str,np.matrix]:
        # Get the ordered list of bodies to iterate over
        body_names = self.topology.get_ordered_body_list()
        # Initialize the states to track velocity & acceleration
        nonlinear_joint_space_forces = self.inverse_dynamics(
            joint_space_accelerations={
                body_name: np.matrix(np.zeros((self.topology.joints[body_name].get_number_degrees_of_freedom())))
                for body_name in body_names
            }
        )
        return nonlinear_joint_space_forces
    
    def forward_dynamics(self, joint_space_forces:typing.Dict[str,np.matrix]) -> typing.Dict[str,np.matrix]:
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

    def step(self, dt:float) -> None:
        joint_space_forces = {}
        body_names = self.topology.get_ordered_body_list()
        for body_name in body_names[1:]:
            for dynamics_module in self.joint_dynamics:
                joint_space_forces[body_name] = dynamics_module(self.topology, body_name)
        joint_space_accelerations = self.forward_dynamics(joint_space_forces)
        for body_name in body_names[1:]:

            A = joint_space_accelerations[body_name]
            if len(self.joint_space_position_history) == 1:
                x0 = self.joint_space_position_history[0][body_name]
                v0 = self.joint_space_velocity_history[0][body_name]
                x1 = x0 + v0*dt + 0.5*A*dt**2
                v1 = v0 + A*dt
                # Update joint position
                self.topology.joints[body_name].set_configuration(x1)
                # Update joint velocity
                self.topology.joints[body_name].set_configuration_d(v1)
            else:
                xn = self.joint_space_position_history[-1][body_name]
                xnminus1 = self.joint_space_position_history[-2][body_name]
                xnplus1 = 2*xn - xnminus1 + A*dt**2
                vnplus1 = (xnplus1 - xnminus1) / (2*dt)
                # Update joint position
                self.topology.joints[body_name].set_configuration(xnplus1)
                # Update joint velocity
                self.topology.joints[body_name].set_configuration_d(vnplus1)
        self.joint_space_position_history.append(self.topology.get_joint_space_positions())
        self.joint_space_velocity_history.append(self.topology.get_joint_space_velocities())
        self.time_history.append(self.time_history[-1]+dt)
        self.topology.update_body_velocities()

        return

    def simulate(self, delta_t:float, dt:float, verbose=False) -> None:
        """
        Runs the simulation for delta_t more seconds.
        """
        if verbose:
            for _ in tqdm.tqdm(range(int(delta_t//dt+1))):
                self.step(dt=dt)
        else:
            for _ in range(int(delta_t//dt+1)):
                self.step(dt=dt)
    
if __name__ == "__main__":
    from pyboatsim.example.example_topology import get_pendulum

    N = 1
    pendulum, pendulum_vis = get_pendulum(N)
    pendulum.joints["Arm 0"].set_configuration(np.matrix([0.9*np.pi/2]))
    sim = Sim(
        topology=pendulum, 
        body_dynamics=[
            Gravity("gravity", -9.81, 2)
        ])

    sim.simulate(10, 0.01, verbose=True)

    plt.scatter(
        sim.time_history, 
        [x["Arm 0"][0,0] for x in sim.joint_space_position_history],
        label="Simulated",
        c="r", marker="x", s=0.5)
    plt.plot(
        sim.time_history, 
        [-0.1*np.pi/2 * np.cos(np.sqrt(9.81) * t) + np.pi/2 for t in sim.time_history],
        label="Expected"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Pendulum Angle (rad)")

    plt.show()
