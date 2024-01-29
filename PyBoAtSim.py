import scipy.integrate as integrate
import numpy as np
import typing as type
import pandas as pd
import matplotlib.pyplot as plt
import tqdm as tqdm

class BoAtSim:
    def __init__(
            self,
            state: type.Dict[str, float] = {
                "t": 0,
                "x": 0,
                "v_boat": 0,
                "alpha": 0,
                "omega": 0.1,
                "R": 0.10,
                "d": 0.05,
                "h": 0.01,
                "N_paddles": 8,
                "rho": 1000,
                "C_paddle": 1.28,
                "C_front": 1.28,
                "dt": 0.01,
                "m": 2,
                "A_front": np.sqrt(0.5) * 0.075 * 0.15,
                "v_water": 0
            }
    ) -> None:
        """
        Initializer
        """
        self._state = state
        self.history = pd.DataFrame(columns=self._state.keys())
        self.history_cache = {}

    def update_state(self, state:type.Dict[str, float]) -> None:
        """
        Update the simulation parameters without having to specify the
        whole `state` dictionary.
        """
        self._state.update(state)


    def calculate_waterwheel_force(self) -> float:
        """
        Integrates the paddle pressure over the surface of each paddle
        to get the force acting on the water wheels.
        """
        paddle_angles = self._state["alpha"] + np.linspace(
            start = 0, 
            stop = 2*np.pi,
            num = self._state["N_paddles"]
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
        # If the section of the wheel is in the water, calculate pressure
        if (
            -np.arccos(self._state["h"] / self._state["R"]) < paddle_angle
            and
            paddle_angle < np.arccos(self._state["h"] / self._state["R"])
            and
            min(abs(self._state["h"]/np.cos(paddle_angle)), self._state["R"]) < l
            and
            l < self._state["R"]
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
            "f_body_drag": 0
        }
        self.update_state(forces)
        total_force = sum(forces.values())
        # Calculate acceleration & update the velocity and position
        a = total_force/self._state["m"]
        self.update_state({"a":a})
        # With everything calculated, append to cache
        self.flush_state_to_cache()
        # Initialize a dict to hold the new sim state
        new_state = self._state

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
    sim = BoAtSim()
    sim.simulate(20, verbose=True)
    plt.plot(sim.history["alpha"]%(2*np.pi), sim.history["f_paddles"], label="paddles")
    # plt.plot(sim.history["t"], sim.history["f_body_drag"], label="body")
    plt.legend()
    plt.show()
    plt.plot(sim.history["t"], sim.history["x"])
    plt.show()

