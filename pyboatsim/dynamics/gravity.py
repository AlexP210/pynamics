import typing

import numpy as np
import trimesh

from pyboatsim.dynamics import DynamicsParent
from pyboatsim.state import State
from pyboatsim.constants import AXES, EPSILON

class MeshGravity(DynamicsParent):
    def __init__(
            self,
            name: str,
            gravity_model_path: str
        ):
        super().__init__(name=name)
        self.dynamics_parameters = {
            "gravity_model_path": gravity_model_path 
        }
        self.gravity_model = trimesh.load(
            file_obj=gravity_model_path, 
            file_type=gravity_model_path.split(".")[-1], 
            force="mesh"
        )

    def required_state_labels(self):
        return [
            "m__boat"
        ]+ [
            f"r_{axis}__boat" for axis in AXES
        ] + [
            f"theta_{axis}__boat" for axis in AXES
        ]
    
    def compute_dynamics(self, state:State, dt:float) -> State:
        # Array representation of position & rotation
        theta = np.array([state[f"theta_{axis}__boat"] for axis in AXES])
        r = np.array([state[f"r_{axis}__boat"] for axis in AXES])

        # Transform the mesh
        translation_matrix = trimesh.transformations.translation_matrix(direction=r)
        if np.linalg.norm(theta) <= EPSILON:
            rotation_matrix = np.eye(4)
        else:
            rotation_matrix = trimesh.transformations.rotation_matrix(
                direction=theta/np.linalg.norm(theta),
                angle=np.linalg.norm(theta))
        transformation_matrix = trimesh.transformations.concatenate_matrices(
                translation_matrix,
                rotation_matrix
        )

        gravity_model_temp:trimesh.Trimesh = self.gravity_model.copy()
        gravity_model_temp.apply_transform(
            matrix=transformation_matrix
        )

        force = np.array([0, 0, -state["m__boat"] * 9.81])
        point_of_application = self.gravity_model.center_mass
        torque = np.cross(point_of_application, force)
        for axis_idx in range(len(AXES)):
            axis = AXES[axis_idx]
            state.set({
                f"f_{axis}__{self.name}": force[axis_idx],
                f"tau_{axis}__{self.name}": torque[axis_idx]
            })

        return state