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
            model_path: str,
            buoyant_model_path: str
        ):
        super().__init__(name=name)
        self.dynamics_parameters = {
            "mesh": trimesh.load(
                file_obj=model_path, 
                file_type=model_path.split(".")[-1], 
                force="mesh"
            ),
            "buoyant_volume": trimesh.load(
                file_obj=buoyant_model_path, 
                file_type=buoyant_model_path.split(".")[-1], 
                force="mesh"
            ),
        }
        self.dynamics_parameters["mesh"].merge_vertices()
        self.dynamics_parameters["mesh"].fix_normals()
        if not self.dynamics_parameters["mesh"].is_watertight:
            raise ValueError("Provided mesh is not watertight.")
        self.dynamics_parameters["buoyant_volume"].merge_vertices()
        self.dynamics_parameters["mesh"].fix_normals()
        if not self.dynamics_parameters["buoyant_volume"].is_watertight:
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
        # print(r)

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

        buoyant_volume:trimesh.Trimesh = self.dynamics_parameters["buoyant_volume"].copy()
        buoyant_volume.apply_transform(
            matrix=transformation_matrix
        )
        submerged:trimesh.Trimesh = buoyant_volume.slice_plane(
            plane_origin=(0,0,0),
            plane_normal=(0,0,-1),
            cap=True
        )
        # Find the submerged volume
        # submerged = self._submerged_volume(buoyant_volume, state, axis=2)
        # Calculate buoyancy
        if submerged.is_empty:
            force = np.array([0, 0, 0])
            torque = np.array([0, 0, 0])
            state["sub_vol"] = 0
        else:
            submerged_hull = submerged
            if not submerged_hull.is_watertight:
                raise ValueError("Convex hull of submerged volume is not watertight.")
            submerged_hull.rezero()
            water_volume = submerged_hull.volume
            water_mass = water_volume * state["rho__water"]
            force = np.array([0, 0, water_mass*9.81])
            point_of_application = submerged_hull.center_mass
            torque = np.cross(point_of_application, force)
            state["sub_vol"] = water_volume
            # force = np.array([0, 0, 0])
            # torque = np.array([0, 0, 0])

        # Update the state dict
        for axis_idx in range(len(AXES)):
            axis = AXES[axis_idx]
            state.set({
                f"f_{axis}__{self.name}": force[axis_idx],
                # f"tau_{axis}__{self.name}": torque[axis_idx]
                f"tau_{axis}__{self.name}": 0
            })

        return state

    def _submerged_volume(self, buoyant_volume, state, axis=2):
        mesh = buoyant_volume
        min_bound, max_bound = mesh.bounds

        if max_bound[axis] <= state["r_z__water"]: return mesh
        elif min_bound[axis] >= state["r_z__water"]: return trimesh.Trimesh()

        min_bound = np.array(min_bound)
        max_bound = np.array(max_bound)

        center = (min_bound + max_bound) / 2

        cropping_box_max_bound = max_bound.copy()
        cropping_box_max_bound[axis] = state["r_z__water"]
        cropping_box_min_bound = min_bound.copy()
        cropping_box_scale = cropping_box_max_bound - cropping_box_min_bound
        cropping_box_center = center.copy()
        cropping_box_center[axis] = min_bound[axis] + 0.5*cropping_box_scale[axis]

        transform = np.eye(4, 4)
        transform[0:3, 3] = cropping_box_center
        cropping_box = trimesh.primitives.Box(extents=cropping_box_scale, transform=transform)
        intersection = trimesh.boolean.intersection([mesh, cropping_box], check_volume=True)

        return intersection
