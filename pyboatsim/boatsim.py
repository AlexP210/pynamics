import typing
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm as tqdm

from pyboatsim.constants import HOME, AXES
from pyboatsim.state import State
from pyboatsim.dynamics import BodyDynamicsParent, JointDynamicsParent, Gravity, JointDamping
from pyboatsim.math import linalg
from pyboatsim.kinematics.topology import Topology, Frame, Body

class Sim:
    def __init__(
            self,
            topology: Topology,
            body_dynamics: typing.Dict[str, BodyDynamicsParent] = {},
            joint_dynamics: typing.Dict[str, JointDynamicsParent] = {},
        ) -> None:
        """Initialize a `Sim`ulation.

        Args:
            topology (Topology): A Topology object representing the initial condition.
            body_dynamics (typing.List[DynamicsParent], optional): A list of BodyDynamics modules to apply forces to the topology. Defaults to [].
            joint_dynamics (typing.List[DynamicsParent], optional): A list of JointDynamics modules to apply forces to the topology. Defaults to [].
        """
        self.body_dynamics:typing.List[BodyDynamicsParent] = body_dynamics #{bd.name: bd for bd in body_dynamics}
        self.joint_dynamics:typing.List[JointDynamicsParent] = joint_dynamics #{jd.name: jd for jd in joint_dynamics}
        self.topology:Topology = topology
        self.joint_space_position_history = []
        self.joint_space_velocity_history = []
        self.time_history = []
        self.data_history:typing.Dict[str:typing.List[float]] = {}
        self._data_collection_callbacks:typing.List[typing.Callable[[Sim], typing.Dict[str, float]]] = []

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
            for name, dynamics_module in self.body_dynamics.items():
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
                body_name: np.matrix(np.zeros((self.topology.joints[body_name].get_configuration_d().size))).T
                for body_name in body_names
            }
        )
        return nonlinear_joint_space_forces
    
    def forward_dynamics(self, joint_space_forces:typing.Dict[str,np.matrix]) -> typing.Dict[str,np.matrix]:
        C = self.topology.vectorify_velocity(self.get_nonlinear_forces())
        H = self.topology.matrixify(self.topology.get_mass_matrix())
        tau = self.topology.vectorify_velocity(joint_space_forces)
        q_dd = np.linalg.inv(H) @ (tau - C)
        body_accelerations = self.topology.calculate_body_accelerations(q_dd)
        joint_space_accelerations = self.topology.dictionarify(q_dd)
        c = self.topology.dictionarify(C)
        t = self.topology.dictionarify(tau)
        return joint_space_accelerations       

    def step(self, dt:float) -> None:

        # Save the current state
        if "Time" not in self.data_history: self.data_history["Time"] = [0,]
        else: self.data_history["Time"].append(self.data_history["Time"][-1]+dt)
        joint_space_positions = self.topology.get_joint_space_positions()
        joint_space_velocities = self.topology.get_joint_space_velocities()

        for joint_name, joint in self.topology.joints.items():
            for dof_idx in range(joint.get_configuration().size):
                position_data_field = f"{joint_name} / Position {dof_idx}"
                position_data = joint_space_positions[joint_name][dof_idx,0]
                self._add_to_data(position_data_field, position_data)
            for dof_idx in range(joint.get_configuration_d().size):
                velocity_data_field = f"{joint_name} / Velocity {dof_idx}"
                velocity_data = joint_space_velocities[joint_name][dof_idx,0]
                self._add_to_data(velocity_data_field, velocity_data)
        
        for body_dynamics_module_name, body_dynamics_module in self.body_dynamics.items():
            for data_field_name, data_value in body_dynamics_module.get_data().items():
                self._add_to_data(f"{body_dynamics_module_name} / {data_field_name}", data_value)

        for joint_dynamics_module_name, joint_dynamics_module in self.joint_dynamics.items():
            for data_field_name, data_value in joint_dynamics_module.get_data().items():
                self._add_to_data(f"{joint_dynamics_module_name} / {data_field_name}", data_value)
        
        # TODO: Update Visualizer to no longer need these so they can be deprecated 
        self.joint_space_position_history.append(self.topology.get_joint_space_positions())
        self.joint_space_velocity_history.append(self.topology.get_joint_space_velocities())
        self.time_history.append(self.time_history[-1]+dt if self.time_history else 0)

        joint_space_forces = {}
        body_names = self.topology.get_ordered_body_list()
        for body_name in body_names[1:]:
            q_shape = self.topology.joints[body_name].get_configuration_d().shape
            joint_space_forces[body_name] = np.matrix(np.zeros(shape=q_shape))
            for dynamics_module_name, dynamics_module in self.joint_dynamics.items():
                joint_space_forces[body_name] += dynamics_module(self.topology, body_name)
        joint_space_accelerations = self.forward_dynamics(joint_space_forces)
        for body_name in body_names[1:]:
            A = joint_space_accelerations[body_name]
            self.topology.joints[body_name].integrate(dt, A)
        # Update the velocities of each body based on the new velocities of each joint
        self.topology.update_body_velocities()
        
        # Update the dynamics modules
        for name, dynamics_module in self.body_dynamics.items(): dynamics_module.update(self.topology, dt)
        for name, dynamics_module in self.joint_dynamics.items(): dynamics_module.update(self.topology, dt)

        return

    def simulate(self, delta_t:float, dt:float, verbose=False) -> None:
        """
        Runs the simulation for delta_t more seconds.
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
            for _ in tqdm.tqdm(range(int(delta_t//dt+1))):
                self.step(dt=dt)
        else:
            for _ in range(int(delta_t//dt+1)):
                self.step(dt=dt)

    def save_data(self, path:str):
        pd.DataFrame(self.data_history).to_csv(path, index=False, sep=",")

    def _add_to_data(self, data_field_name, data_value):
        if data_field_name in self.data_history:
            self.data_history[data_field_name].append(data_value)
        else:
            self.data_history[data_field_name] = [data_value,]
    
if __name__ == "__main__":
    from pyboatsim.example.example_topology import get_pendulum

    N = 1
    eps = 0.01
    g = -9.81
    damp = 0.2
    pendulum, pendulum_vis = get_pendulum(N)
    pendulum.joints["Arm 0"].set_configuration(np.matrix([(1-eps)*np.pi/2]))
    sim = Sim(
        topology=pendulum, 
        body_dynamics=[Gravity("gravity", g, 2),],
        joint_dynamics=[JointDamping("damping", damp),]
    )

    sim.simulate(30, 0.01, verbose=True)
    
    pendulum_vis.add_sim_data(sim)
    pendulum_vis.animate(f"Damping_Test_{N}.mp4")

    plt.scatter(
        sim.time_history, 
        [x["Arm 0"][0,0] for x in sim.joint_space_position_history],
        label="Simulated",
        c="r", marker="x", s=0.5)
    plt.plot(
        sim.time_history, 
        [-eps*np.pi/2 * np.exp(-damp/2 * t) * np.cos(np.sqrt(-g) * t) + np.pi/2 for t in sim.time_history],
        label="Expected"
    )
    plt.plot(
        sim.time_history, 
        [eps*np.pi/2 * np.exp(-damp/2 * t) + np.pi/2 for t in sim.time_history],
        c="orange"
    )
    plt.plot(
        sim.time_history, 
        [-eps*np.pi/2 * np.exp(-damp/2 * t) + np.pi/2 for t in sim.time_history],
        label="Expected Envelope",
        c="orange"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Pendulum Angle (rad)")
    plt.legend()

    plt.show()
