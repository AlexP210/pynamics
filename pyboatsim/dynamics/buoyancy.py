import typing

import numpy as np
import trimesh

from pyboatsim.dynamics import DynamicsParent
from pyboatsim.state import State
from pyboatsim.constants import AXES, EPSILON
from pyboatsim.topology import Topology

class MeshBuoyancy(DynamicsParent):
    def __init__(
            self,
            name: str,
            buoyancy_model_path: str,
            fluid_density: float,
            fluid_height: float
        ):
        super().__init__(name=name)
        self.dynamics_parameters = {
            "buoyancy_model_path": buoyancy_model_path,
            "fluid_density": fluid_density,
            "fluid_height": fluid_height
        }
        self.buoyancy_model = trimesh.load(
            file_obj=buoyancy_model_path, 
            file_type=buoyancy_model_path.split(".")[-1], 
            force="mesh"
        )
        self.buoyancy_model.merge_vertices()
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
    
    def compute_dynamics(self, state:State, topology:Topology, dt:float) -> State:

        # Array representation of position & rotation of root body inertial frame
        theta = np.array([state[f"theta_{axis}__boat"] for axis in AXES])
        r = np.array([state[f"r_{axis}__boat"] for axis in AXES])
        c = np.array([state[f"c_{axis}__boat"] for axis in AXES])

        # Get transformation from World frame to topology com frame
        T_root_com = trimesh.transformations.translation_matrix(direction=c)
        T_com_root = trimesh.transformations.translation_matrix(direction=-c)
        T_world_root = trimesh.transformations.translation_matrix(direction=r)
        if np.linalg.norm(theta) <= EPSILON:
            C_world_com = np.eye(4)
        else:
            C_world_com = trimesh.transformations.rotation_matrix(
                direction=theta/np.linalg.norm(theta),
                angle=np.linalg.norm(theta)
            )
        transformation_matrix = trimesh.transformations.concatenate_matrices(
                T_root_com,
                T_world_root,
                C_world_com,
                T_com_root
        )

        # Copy of the buoyancy model to move around
        buoyancy_model_temp:trimesh.Trimesh = self.buoyancy_model.copy()
        # Transform the mesh
        buoyancy_model_temp.apply_transform(
            matrix=transformation_matrix
        )

        submerged:trimesh.Trimesh = buoyancy_model_temp.slice_plane(
            plane_origin=(0,0,0),
            plane_normal=(0,0,-1),
            cap=True
        )

        # Matrix representations of position & rotation
        theta_m = np.matrix(theta).T
        r_m = np.matrix(r).T
        c_m = np.matrix([
            [state["c_x__boat"],],
            [state["c_y__boat"],],
            [state["c_z__boat"],],
        ])

        # Calculate buoyancy
        if submerged.is_empty:
            force__worldframe = np.matrix([0, 0, 0]).T
            torque__comframe = np.matrix([0, 0, 0]).T
            state[f"{self.name}__submerged_volume"] = 0
        else:
            if not submerged.is_watertight:
                raise ValueError("Convex hull of submerged volume is not watertight.")
            water_volume = submerged.volume
            state[f"{self.name}__submerged_volume"] = water_volume
            water_mass = water_volume * state["rho__water"]
            force__worldframe = np.matrix([0, 0, water_mass*9.81]).T
            point_of_application__worldframe = np.matrix(submerged.center_mass).T
            force__bodyframe = np.dot(C_world_com.T[0:3, 0:3], force__worldframe)
            force__comframe = force__bodyframe
            point_of_application__bodyframe = np.dot(C_world_com.T[0:3, 0:3], point_of_application__worldframe - r_m)
            point_of_application__comframe = point_of_application__bodyframe - c_m
            torque__comframe = np.cross(
                point_of_application__comframe.T,
                force__comframe.T
            ).T

        # Update the state dict
        for axis_idx in range(len(AXES)):
            axis = AXES[axis_idx]
            state.set({
                f"f_{axis}__{self.name}": force__worldframe[axis_idx,0],
                f"tau_{axis}__{self.name}": torque__comframe[axis_idx,0]
            })

        return state