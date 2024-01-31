import typing as type
import argparse
import os

import scipy.integrate as integrate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm as tqdm

import Constants

class BoAtSim:
    def __init__(
            self,
            state: type.Dict[str, float] = {
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
            }) -> None:
        """
        Initializer
        """
        self._state = state
        self.history = pd.DataFrame(columns=self._state.keys())
        self.history_cache = {}

    def load_state(self, name, state_database: pd.DataFrame):
        """
        Selects the state with index `name` from `state_database`.
        `state_database` can be a database of initial states or a previous
        sim output.
        """
        # Pandas exports the dataframe as {col: {index: value}} format, even if
        # there's only one index value in the dataframe. We need {col: value} 
        # for the index selected
        state_database_as_dict = state_database.to_dict()
        state_dict = {col: state_database_as_dict[col][name] 
                      for col in state_database_as_dict.keys()}
        self.update_state(state_dict)


    def update_state(self, state:type.Dict[str, float]) -> None:
        """
        Update the simulation parameters without having to specify the
        whole `state` dictionary.
        """
        self._state.update(state)

    def get_state(self, keys:type.List[str]=None) -> type.Dict[str, float]:
        """
        Returns (a subset of) the state.
        """
        if keys is None: keys = self._state.keys()
        return {key: self._state[key] for key in keys}


    def calculate_waterwheel_force(self) -> float:
        """
        Integrates the paddle pressure over the surface of each paddle
        to get the force acting on the water wheels.
        """
        paddle_angles = self._state["alpha"] + np.linspace(
            start = -np.pi, 
            stop = np.pi,
            num = self._state["N_paddles"],
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
        integrand = lambda l: self._state["d"] * self.calculate_paddle_pressure(
            paddle_angle=paddle_angle,
            l=l
        )
        return integrate.quad(
            func=integrand,
            a=min(abs(self._state["h"]/np.cos(paddle_angle)), self._state["R"]),
            b=float(self._state["R"])
        )[0]
    
    def calculate_body_drag_force(self):
        """
        Calculates the drag force acting on the body of the Bo-At
        """
        factors = [
            np.sign(self._state["v_water"] - self._state["v_boat"]),
            0.5*self._state["C_front"]*self._state["rho"],
            self._state["A_front"],
            (self._state["v_water"]-self._state["v_boat"])**2
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
            -np.arccos(self._state["h"] / self._state["R"]) <= paddle_angle
            and
            paddle_angle <= np.arccos(self._state["h"] / self._state["R"])
            and
            min(abs(self._state["h"]/np.cos(paddle_angle)), self._state["R"]) <= l
            and
            l <= self._state["R"]
        ):
            cos_a = np.cos(paddle_angle)
            factors = [
                self._state["C_paddle"] * self._state["rho"] / 2,
                np.sqrt(
                        (self._state["v_water"] - self._state["v_boat"])**2 +
                        2*(self._state["v_water"] - self._state["v_boat"])*self._state["omega"]*cos_a*l +
                        self._state["omega"]**2 * l**2
                    ),
                (self._state["v_water"] - self._state["v_boat"]) * cos_a + self._state["omega"] * l,
                cos_a
            ]
            return np.prod(factors)
        # If the section of the wheel is not in the water, no presure
        else:
            return 0
        
    def flush_state_to_cache(self):
        for key, value in self._state.items():
            if key in self.history_cache: self.history_cache[key].append(value)
            else: self.history_cache[key] = [value,]

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
        a = total_force/self._state["m"]
        self.update_state(state={"a":a})
        # With everything calculated, append to cache
        self.flush_state_to_cache()
        # Initialize a dict to hold the new sim state
        new_state = self.get_state()

        # Update state
        new_state["v_boat"] += a*self._state["dt"]
        new_state["x"] += 0.5*(new_state["v_boat"]+self._state["v_boat"])*self._state["dt"]
        # Update the angle of the water wheels
        new_state["alpha"] += self._state["omega"] * self._state["dt"]
        # Update the sim time
        new_state["t"] += self._state["dt"]
        # Set the state
        self._state = new_state

    def simulate(self, delta_t:float, verbose=False):
        """
        Runs the simulation for delta_t more seconds.
        """
        if verbose:
            for _ in tqdm.tqdm(range(int(delta_t//self._state["dt"]))):
                self.step()
        else:
            for _ in range(int(delta_t//self._state["dt"])):
                self.step()
        self.flush_state_to_cache()
        self.history = pd.DataFrame(self.history_cache)

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
        Constants.HOME, "Outputs", 
        args.initial_state_name+".csv")

    # Assemble the sim
    sim = BoAtSim()
    sim.load_state(
        name=args.initial_state_name,
        state_database=pd.read_csv("InitialConditions.csv", index_col="Name")
    )
    sim.update_state(state={"dt":args.time_step})

    # Run the sim
    sim.simulate(delta_t=float(args.duration), verbose=True)

    # Save the outputs
    sim.history.to_csv(
        path_or_buf=args.output,
        sep=",",
        index="Name")