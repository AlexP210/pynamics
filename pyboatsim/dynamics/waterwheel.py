import typing

import numpy as np
import scipy.integrate as integrate

from dynamics import DynamicsParent
from pyboatsim.state import State

class WaterWheel(DynamicsParent):
    def __init__(
            self,
            wheel_radius: float,
            paddle_width: float,
            wheel_hub_height: float,
            number_of_paddles_per_wheel: float,
            paddle_drag_coefficient: float,
            mass: float,
            side: str,
        ):
        self.dynamics_parameters = {
            "radius": wheel_radius,
            "paddle_width": paddle_width,
            "height": wheel_hub_height,
            "number_of_paddles": number_of_paddles_per_wheel,
            "drag_coefficient": paddle_drag_coefficient,
        }
        self.name = f"waterwheel_{side}"

    def required_state_labels(self):
        return [
            f"alpha_{self.name}",
            f"omega_{self.name}",
            "rho",
            "v_boat",
            "v_water"
        ]
    
    def compute_dynamics(self, state:State):
        return self._calculate_waterwheel_force(
            alpha = state[f"alpha_{self.name}"],
            rho=state["rho"],
            v_water=state["v_water"],
            v_boat=state["v_boat"],
            omega=state[f"omega_{self.name}"]
        )

    def _calculate_paddle_pressure(
            self, 
            paddle_angle:float, 
            l:float, 
            rho:float,
            v_water:float,
            v_boat:float,
            omega:float
        ) -> float:
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
            -np.arccos(self.dynamics_parameters["height"] / self.dynamics_parameters["radius"]) <= paddle_angle
            and
            paddle_angle <= np.arccos(self.dynamics_parameters["height"] / self.dynamics_parameters["radius"])
            and
            min(abs(self.dynamics_parameters["height"]/np.cos(paddle_angle)), self.dynamics_parameters["radius"]) <= l
            and
            l <=  self.dynamics_parameters["radius"]
        ):
            cos_a = np.cos(paddle_angle)
            factors = [
                self.dynamics_parameters["drag_coefficient"] * rho / 2,
                np.sqrt(
                        (v_water - v_boat)**2 +
                        2*(v_water - v_boat)*omega*cos_a*l +
                        omega**2 * l**2
                    ),
                (v_water - v_boat) * cos_a + omega * l,
                cos_a
            ]
            return np.prod(factors)
        # If the section of the wheel is not in the water, no presure
        else:
            return 0

    def _calculate_paddle_force(
            self, 
            paddle_angle: float,
            rho:float,
            v_water:float,
            v_boat:float,
            omega:float
        ) -> float:
        """
        Calculates the force acting on a paddle.
        """
        integrand = lambda l: self.state["d"] * self._calculate_paddle_pressure(
            paddle_angle=paddle_angle,
            l=l,
            rho=rho,
            v_water=v_water,
            v_boat=v_boat,
            omega=omega
        )
        return integrate.quad(
            func=integrand,
            a=min(abs(self.state["h"]/np.cos(paddle_angle)), self.state["R"]),
            b=float(self.state["R"])
        )[0]

    def _calculate_waterwheel_force(
            self,
            alpha:float,
            rho:float,
            v_water:float,
            v_boat:float,
            omega:float
        ) -> float:
        """
        Integrates the paddle pressure over the surface of each paddle
        to get the force acting on the water wheels.
        """
        paddle_angles = alpha + np.linspace(
            start = -np.pi, 
            stop = np.pi,
            num = self.dynamics_parameters["number_of_paddles"],
            endpoint=False
        )
        paddle_forces = [
            self._calculate_paddle_force(
                paddle_angle=paddle_angle,
                rho=rho,
                v_water=v_water,
                v_boat=v_boat,
                omega=omega
            )
            for paddle_angle in paddle_angles
        ]
        return sum(paddle_forces)
