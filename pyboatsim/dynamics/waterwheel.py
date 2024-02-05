import typing

from dynamics import Dynamics

class WaterWheel(Dynamics):
    def __init__(
            self,
            wheel_radius: float,
            paddle_width: float,
            wheel_hub_height: float,
            number_of_paddles_per_wheel: float,
            paddle_drag_coefficient: float,
            mass: float,
            side: float,
        ):
        self.dynamics_parameters = {
            "radius": wheel_radius,
            "paddle_width": paddle_width,
            "height": wheel_hub_height,
            "number_of_paddles": number_of_paddles_per_wheel,
            "drag_coefficient": paddle_drag_coefficient,
            "mass": mass,
        }
        self.NAME = f"waterwheel_{side}"

    def required_state_labels(self):
        return [
            f"alpha_{self.NAME}",
            f"omega_{self.NAME}",
            "rho",
            "v_boat",
            "v_water"
        ]
    
    def compute_dynamics(self, state):
        pass 
