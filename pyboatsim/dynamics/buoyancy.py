import typing

import numpy as np
import trimesh

from pyboatsim.dynamics import BodyDynamicsParent
from pyboatsim.state import State
from pyboatsim.constants import AXES, EPSILON
from pyboatsim.kinematics.topology import Topology

from pyboatsim.math.linalg import R3_cross_product_matrix

class Buoyancy(BodyDynamicsParent):
    def __init__(
            self,
            buoyancy_models: typing.Dict[str, trimesh.Trimesh],
            direction: int = 2,
            fluid_level: float = 0,
            fluid_density: float = 1000,
            g: float = -9.81
        ):
        super().__init__(body_names=buoyancy_models.keys())
        self.direction = direction
        self.fluid_level = fluid_level
        self.fluid_density = fluid_density
        self.buoyancy_models = buoyancy_models
        self.g = g
        for name, model in self.buoyancy_models.items():
            if not model.is_watertight: raise ValueError(f"Buoyant volume mesh for {name} is not watertight.")
    
    def compute_dynamics(self, topology:Topology, body_name:str) -> typing.Tuple[np.matrix, np.matrix]:
        
        # Copy of the buoyancy model to move around
        buoyancy_model_temp:trimesh.Trimesh = self.buoyancy_models[body_name].copy()
        buoyancy_model_temp.apply_transform(
            matrix=np.array(topology.get_transform("World", "Identity", body_name, "Identity"))
        )


        plane_origin = [0,0,0]
        plane_normal = [0,0,0]
        plane_origin[self.direction] = self.fluid_level
        plane_normal[self.direction] = -1
        submerged:trimesh.Trimesh = buoyancy_model_temp.slice_plane(
            plane_origin=plane_origin,
            plane_normal=plane_normal,
            cap=True
        )
        force = np.matrix([0, 0, 0]).T
        point_of_application = np.matrix([0, 0, 0]).T
        if not submerged.is_empty:
            water_volume = submerged.volume
            water_mass = water_volume * self.fluid_density
            force[self.direction,0] = -water_mass*self.g
            point_of_application = np.matrix(submerged.center_mass).T
        return [(force, point_of_application),]
