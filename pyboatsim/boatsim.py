import typing
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm as tqdm

from pyboatsim.constants import HOME, AXES
from pyboatsim.state import State
from pyboatsim.dynamics import DynamicsParent, WaterWheel, SimpleBodyDrag, ConstantForce, MeshBuoyancy, MeshGravity
from pyboatsim.math import linalg

class BoAtSim:
    def __init__(
            self,
            state: State,
            dynamics: typing.List[DynamicsParent]
        ) -> None:
        """
        Initializer
        """
        self.state = state
        self.history = []
        self.dynamics = dynamics
        self.dynamics_names = [
            dynamics_module.name 
            for dynamics_module in self.dynamics
        ]
        self.required_labels = [
            f"r_{axis}__boat" for axis in AXES
        ] + [
            f"v_{axis}__boat" for axis in AXES
        ] + [
            f"a_{axis}__boat" for axis in AXES
        ] + [
            f"theta_{axis}__boat" for axis in AXES
        ] + [
            f"omega_{axis}__boat" for axis in AXES
        ] + [
            f"alpha_{axis}__boat" for axis in AXES
        ] + [
            f"m__boat"
        ] + [
            f"I_{axis}{axis}__boat" for axis in AXES
        ] + [
            f"c_{axis}__boat" for axis in AXES
        ] + [
            "t"
        ]

    def _compute_accelerations(self):
        # Force and Moment
        F = np.matrix([self.state[f"f_{axis}__total"] for axis in AXES]).T
        T = np.matrix([self.state[f"tau_{axis}__total"] for axis in AXES]).T
        FT = np.block([
            [F,],
            [T,],
        ])
        # Angular velocity
        w = np.matrix([self.state[f"omega_{axis}__boat"] for axis in AXES]).T
        w_x = linalg.cross_product_matrix(w)
        # Mass properties
        I_cm = np.array([
            [self.state["I_xx__boat"], self.state["I_xy__boat"],  self.state["I_xz__boat"]],
            [self.state["I_yx__boat"], self.state["I_yy__boat"],  self.state["I_yz__boat"]],
            [self.state["I_zx__boat"], self.state["I_zy__boat"],  self.state["I_zz__boat"]],
        ])
        m = self.state["m__boat"]
        c = np.matrix([self.state[f"c_{axis}__boat"] for axis in AXES]).T
        c_x = linalg.cross_product_matrix(c)

        # Identity matrix
        I3 = np.eye(3)

        # Assemble the matrices from https://en.wikipedia.org/wiki/Newton%E2%80%93Euler_equations#Any_reference_frame
        A = np.block([
            [m*I3, -m*c_x],
            [m*c_x, I_cm - m*c_x@c_x]
        ])
        B = np.block([
            [m*w_x@w_x@c,],
            [w_x@(I_cm - m*c_x@c_x)@w]
        ])

        accelerations = np.linalg.inv(A)*(FT - B)
        
        for idx, axis in enumerate(AXES):
            self.state[f"a_{axis}__boat"] = accelerations[idx, 0]
            self.state[f"alpha_{axis}__boat"] = accelerations[2+idx, 0]

        return 

    def step(self, dt):
        """
        Steps the simulation by `self._state["dt"]`.
        """
        # Apply the dynamics on the state, adds forces, torques, and other
        # intermediate values calculated by dynamics modules based on the
        # current state.
        for dynamics_module in self.dynamics:
            self.state = dynamics_module(self.state, dt)
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
        for axis in AXES:
            # Update the velocities
            next_state[f"v_{axis}__boat"] += self.state[f"a_{axis}__boat"]*dt
            next_state[f"omega_{axis}__boat"] += self.state[f"alpha_{axis}__boat"]*dt
            # Update the positions
            next_state[f"r_{axis}__boat"] += 0.5*(self.state[f"v_{axis}__boat"] + next_state[f"v_{axis}__boat"])*dt
            next_state[f"theta_{axis}__boat"] += 0.5*(self.state[f"omega_{axis}__boat"] + next_state[f"omega_{axis}__boat"])*dt
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
        pd.DataFrame.from_dict(self.history).to_csv(file_path)

if __name__ == "__main__":

    # Assemble the sim
    sim = BoAtSim(
        state=State(
            state_dictionary={
            "t": 0,
            "r_x__boat": 0, 
            "r_y__boat": 0,
            "r_z__boat": 0,
            "r_z__water": 0,
            "v_x__boat": 0,
            "v_y__boat": 0, 
            "v_z__boat": 0,
            "a_x__boat": 0, 
            "a_y__boat": 0, 
            "a_z__boat": 0, 
            "theta_x__boat": 0, 
            "theta_y__boat": 0, 
            "theta_z__boat": 0, 
            "omega_x__boat": 0, 
            "omega_y__boat": 0, 
            "omega_z__boat": 0,
            "alpha_x__boat": 0, 
            "alpha_y__boat": 0, 
            "alpha_z__boat": 0,
            "m__boat": 1000,
            "I_xx__boat": 1,
            "I_xy__boat": 0,
            "I_xz__boat": 0,
            "I_yx__boat": 0,
            "I_yy__boat": 1,
            "I_yz__boat": 0,
            "I_zx__boat": 0,
            "I_zy__boat": 0,
            "I_zz__boat": 1,
            "c_x__boat": 1,
            "c_y__boat": 1,
            "c_z__boat": 1,
            "rho__water": 1000,
            "v_x__water": 0,
            "v_y__water": 0, 
            "v_z__water": 0,
            "gamma__waterwheel": 0,
            "gammadot__waterwheel": 0.01,
        }), 
        dynamics=[
            MeshBuoyancy(
                name="buoyancy", 
                buoyancy_model_path="/home/alex/Projects/PyBoAtSim/models/cup/cup_boundary.obj"
            ),
            MeshGravity(
                name="gravity", 
                gravity_model_path="/home/alex/Projects/PyBoAtSim/models/cup/cup.obj"
            ),
        ]
    )

    # Run the sim
    sim.simulate(delta_t=20, dt=0.001, verbose=True)
    data = pd.DataFrame.from_dict(sim.history)

    # Plot the results
    plt.plot(data["t"], data["f_z__gravity"], label="f_z__gravity")
    plt.plot(data["t"], data["f_z__buoyancy"], label="f_z__buoyancy")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.legend()
    plt.show()

    plt.plot(data["t"], data["buoyancy__submerged_volume"], label="submerged_volume")
    plt.xlabel("Time (s)")
    plt.ylabel("Submerged Volume (m^3)")
    plt.legend()
    plt.show()

    plt.plot(data["t"], data["r_z__boat"], label="r_z__boat")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.show()