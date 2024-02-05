import typing
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm as tqdm

import constants
from state import State
# from dynamics.dynamics import Dynamics
# from dynamics.bodydrag import BodyDrag
import dynamics

class BoAtSim:
    def __init__(
            self,
            state: State,
            dynamics: typing.List[Dynamics]
            ) -> None:
        """
        Initializer
        """
        self.state = state
        self.history = []
        self.dynamics = dynamics
              
    def step(self):
        """
        Steps the simulation by `self._state["dt"]` using forward euler.
        """
        # Calculate the force acting on us during this dt
        forces = {
            f"force_{dynamic.name}": dynamic(self.state)
            for dynamic in self.dynamics
        }
        self.state.set(partial_state_dictionary=forces)
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

    # Assemble the sim
    sim = BoAtSim(state=State(), dynamics=BodyDrag(0.1, 1.28))
    # sim.state.load(
    #     name=args.initial_state_name,
    #     state_database=pd.read_csv("InitialConditions.csv", index_col="Name")
    # )
    # sim.state.set(state={"dt":args.time_step})

    # Run the sim
    sim.simulate(delta_t=10, verbose=True)

    # # Save the outputs
    # sim.history.to_csv(
    #     path_or_buf=args.output,
    #     sep=",",
    #     index="Name")