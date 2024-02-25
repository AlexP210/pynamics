import typing

import numpy as np
import trimesh

from pyboatsim.dynamics import DynamicsParent
from pyboatsim.state import State
from pyboatsim.constants import AXES

class MeshGravity(DynamicsParent):
    def __init__(
            self,
            name: str,
            model_path: str
        ):
        super().__init__(name=name)
        self.dynamics_parameters = {
            "mesh": trimesh.load(
                file_obj=model_path, 
                file_type=model_path.split(".")[-1], 
                force="mesh"
            )
        }

    def required_state_labels(self):
        return ["m__boat"]
    
    def compute_dynamics(self, state:State, dt:float) -> State:
        force = np.array([0, 0, -state["m__boat"] * 9.81])
        point_of_application = self.dynamics_parameters["mesh"].center_mass
        torque = np.cross(point_of_application, force)
        for axis_idx in range(len(AXES)):
            axis = AXES[axis_idx]
            state.set({
                f"f_{axis}__{self.name}": force[axis_idx],
                f"tau_{axis}__{self.name}": torque[axis_idx]
            })

        return state