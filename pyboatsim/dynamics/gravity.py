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
        c = np.array([state[f"c_{axis}__boat"] for axis in AXES])

        # Copy of the buoyancy model to move around
        gravity_model_temp:trimesh.Trimesh = self.gravity_model.copy()
        # Transform it to the CM frame
        c_translation_matrix = trimesh.transformations.translation_matrix(direction=-c)
        gravity_model_temp.apply_transform(c_translation_matrix)

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

        gravity_model_temp.apply_transform(
            matrix=transformation_matrix
        )
        gravity_model_temp.apply_transform(-c_translation_matrix)
        
        # Matrix representations of position & rotation
        theta_m = np.matrix(theta).T
        r_m = np.matrix(r).T

        force__worldframe = np.matrix([0, 0, -state["m__boat"] * 9.81]).T
        force__bodyframe = np.dot(rotation_matrix.T[0:3, 0:3], force__worldframe)
        force__comframe = force__bodyframe
        point_of_application__bodyframe = np.matrix([
            [state["c_x__boat"],],
            [state["c_y__boat"],],
            [state["c_z__boat"],],
        ])
        point_of_application__comframe = np.matrix([
            [0,],
            [0,],
            [0,],
        ])
        torque__comframe = np.cross(
            force__comframe.T,
            point_of_application__comframe.T
        ).T

        for axis_idx in range(len(AXES)):
            axis = AXES[axis_idx]
            state.set({
                f"f_{axis}__{self.name}": force__worldframe[axis_idx,0],
                f"tau_{axis}__{self.name}": torque__comframe[axis_idx,0]
            })

        return state