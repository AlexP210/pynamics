import typing as type
import argparse
import os

import scipy.integrate as integrate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm as tqdm

import constants
from state import State

class BoAtSim:
    def __init__(
            self,
            state: State = State(
                state_dictionary= {
                    "t": 0,
                    "x": 0,
                    "v_boat": 0,
                    "alpha": 0,
                    "omega": 2*np.pi,
                    "R": 0.10,
                    "d": 0.05,
                    "h": 0.05,
                    "N_paddles": 8,
                    "rho": 1000,
                    "C_paddle": 1.28,
                    "C_front": 1.28,
                    "m": 2,
                    # "A_front": np.sqrt(0.5) * 0.075 * 0.15,
                    "A_front": np.sqrt(0.5) * 0.005 * 0.15,
                    "v_water": 0.0,
                    "dt": 0.01,
                })
            ) -> None:
        """
        Initializer
        """
        self.state = state
        self.history = []

    def calculate_waterwheel_force(self) -> float:
        """
        Integrates the paddle pressure over the surface of each paddle
        to get the force acting on the water wheels.
        """
        paddle_angles = self.state["alpha"] + np.linspace(
            start = -np.pi, 
            stop = np.pi,
            num = self.state["N_paddles"],
            endpoint=False
        )
        paddle_forces = [
            self.calculate_paddle_force(paddle_angle=paddle_angle)
            for paddle_angle in paddle_angles
        ]
        return 2 * sum(paddle_forces)
          
    def calculate_paddle_force(self, paddle_angle: float) -> float:
        """
        Calculates the force acting on a paddle.
        """
        integrand = lambda l: self.state["d"] * self.calculate_paddle_pressure(
            paddle_angle=paddle_angle,
            l=l
        )
        return integrate.quad(
            func=integrand,
            a=min(abs(self.state["h"]/np.cos(paddle_angle)), self.state["R"]),
            b=float(self.state["R"])
        )[0]
    
    def calculate_body_drag_force(self):
        """
        Calculates the drag force acting on the body of the Bo-At
        """
        factors = [
            np.sign(self.state["v_water"] - self.state["v_boat"]),
            0.5*self.state["C_front"]*self.state["rho"],
            self.state["A_front"],
            (self.state["v_water"]-self.state["v_boat"])**2
        ]
        return np.prod(factors)

    def calculate_paddle_pressure(self, paddle_angle: float, l: float) -> float:
        """
        Calculates the horizontal pressure on the paddle at a length of l from
        the hub of the water wheel.
        """
        # Convert the paddle angle from (-inf, inf) to [-pi, pi) to match the
        # convention of the dynamics formulation
        paddle_angle = paddle_angle%(2*np.pi)
        if paddle_angle >= np.pi: paddle_angle -= 2*np.pi
        # If the section of the wheel is in the water, calculate pressure
        if (
            -np.arccos(self.state["h"] / self.state["R"]) <= paddle_angle
            and
            paddle_angle <= np.arccos(self.state["h"] / self.state["R"])
            and
            min(abs(self.state["h"]/np.cos(paddle_angle)), self.state["R"]) <= l
            and
            l <= self.state["R"]
        ):
            cos_a = np.cos(paddle_angle)
            factors = [
                self.state["C_paddle"] * self.state["rho"] / 2,
                np.sqrt(
                        (self.state["v_water"] - self.state["v_boat"])**2 +
                        2*(self.state["v_water"] - self.state["v_boat"])*self.state["omega"]*cos_a*l +
                        self.state["omega"]**2 * l**2
                    ),
                (self.state["v_water"] - self.state["v_boat"]) * cos_a + self.state["omega"] * l,
                cos_a
            ]
            return np.prod(factors)
        # If the section of the wheel is not in the water, no presure
        else:
            return 0
        
    def step(self):
        """
        Steps the simulation by `self._state["dt"]` using forward euler.
        """
        # Calculate the force acting on us during this dt
        forces = {
            "f_paddles": self.calculate_waterwheel_force(),
            "f_body_drag": self.calculate_body_drag_force()
        }
        self.update_state(state=forces)
        total_force = sum(forces.values())
        # Calculate acceleration & update the velocity and position
        a = total_force/self.state["m"]
        self.update_state(state={"a":a})
        # With everything calculated, append to cache
        self.history.append(self.state.copy())
        # Initialize a state to hold the new sim state
        new_state = self.state.copy()
        # Update state
        new_state["v_boat"] += a*self.state["dt"]
        new_state["x"] += 0.5*(new_state["v_boat"]+self.state["v_boat"])*self.state["dt"]
        # Update the angle of the water wheels
        new_state["alpha"] += self.state["omega"] * self.state["dt"]
        # Update the sim time
        new_state["t"] += self.state["dt"]
        # Set the state
        self.state = new_state

    def simulate(self, delta_t:float, verbose=False):
        """
        Runs the simulation for delta_t more seconds.
        """
        if verbose:
            for _ in tqdm.tqdm(range(int(delta_t//self.state["dt"]))):
                self.step()
        else:
            for _ in range(int(delta_t//self.state["dt"])):
                self.step()
        self.flush_state_to_cache()
        self.history = pd.DataFrame(self.history_cache)
    
    def save_history(self, file_path:str):
        pd.DataFrame.from_dict(self.history).to_csv(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyBoAtSim is a simulator for the Bo-At.')
    parser.add_argument('-i', '--initial_state_name', 
                        dest='initial_state_name',
                        help='Name of the initial state from InitialConditions.csv', 
                        default='Mk0_StillWater_1rad',
                        metavar='INIT')
    parser.add_argument('-d', '--duration', 
                        dest='duration',
                        help='Length of time to simulate.', 
                        default=60,
                        metavar='DUR')
    parser.add_argument('-dt', '--time_step', 
                        dest='time_step',
                        help='Time step to use.', 
                        default=0.01,
                        metavar='DT')    
    parser.add_argument('-o', '--output', 
                        dest='output',
                        help='Path to write the sim output to. If None use `Outputs/<initial_state_name>.csv`', 
                        default=None,
                        metavar='OUT')
    # Parse the arguments
    args = parser.parse_args()
    if args.output is None: args.output = os.path.join(
        constants.HOME, "Outputs", 
        args.initial_state_name+".csv")

    # Assemble the sim
    sim = BoAtSim()
    sim.state.load(
        name=args.initial_state_name,
        state_database=pd.read_csv("InitialConditions.csv", index_col="Name")
    )
    sim.state.set(state={"dt":args.time_step})

    # Run the sim
    sim.simulate(delta_t=float(args.duration), verbose=True)

    # Save the outputs
    sim.history.to_csv(
        path_or_buf=args.output,
        sep=",",
        index="Name")