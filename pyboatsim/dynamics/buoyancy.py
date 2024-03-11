import typing

import numpy as np
import trimesh

from pyboatsim.dynamics import DynamicsParent
from pyboatsim.state import State
from pyboatsim.constants import AXES, EPSILON

class MeshBuoyancy(DynamicsParent):
    def __init__(
            self,
            name: str,
            buoyancy_model_path: str
        ):
        super().__init__(name=name)
        self.dynamics_parameters = {
            "buoyancy_model_path": buoyancy_model_path
        }
        self.buoyancy_model = trimesh.load(
            file_obj=buoyancy_model_path, 
            file_type=buoyancy_model_path.split(".")[-1], 
            force="mesh"
        )
        self.buoyancy_model.merge_vertices()
        # self.buoyancy_model.fix_normals()
        # self.buoyancy_model.subdivide_to_size(
        #     max_edge=0.01,
        #     max_iter=10
        # )
        if not self.buoyancy_model.is_watertight:
            raise ValueError("Buoyant volume mesh is not watertight.")

    def required_state_labels(self):
        return [
                "r_z__water",
            ] + [
                f"r_{axis}__boat" for axis in AXES
            ] + [
                f"theta_{axis}__boat" for axis in AXES
            ] + [
                "rho__water"
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

        buoyancy_model_temp:trimesh.Trimesh = self.buoyancy_model.copy()
        buoyancy_model_temp.apply_transform(
            matrix=transformation_matrix
        )
        submerged:trimesh.Trimesh = buoyancy_model_temp.slice_plane(
            plane_origin=(0,0,0),
            plane_normal=(0,0,-1),
            cap=True
        )

        # Calculate buoyancy
        if submerged.is_empty:
            force = np.array([0, 0, 0])
            torque = np.array([0, 0, 0])
            state[f"{self.name}__submerged_volume"] = 0
        else:
            if not submerged.is_watertight:
                raise ValueError("Convex hull of submerged volume is not watertight.")
            # submerged_hull.rezero()
            water_volume = submerged.volume
            state[f"{self.name}__submerged_volume"] = water_volume
            water_mass = water_volume * state["rho__water"]
            force = np.array([0, 0, water_mass*9.81])
            point_of_application = submerged.center_mass
            torque = np.cross(point_of_application, force)
            # force = np.array([0, 0, 0])
            # torque = np.array([0, 0, 0])

        # Update the state dict
        for axis_idx in range(len(AXES)):
            axis = AXES[axis_idx]
            state.set({
                f"f_{axis}__{self.name}": force[axis_idx],
                f"tau_{axis}__{self.name}": torque[axis_idx]
                # f"tau_{axis}__{self.name}": 0
            })

        return state