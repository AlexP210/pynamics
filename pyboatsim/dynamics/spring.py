import typing

import numpy as np
import trimesh

from pyboatsim.dynamics import BodyDynamicsParent
from pyboatsim.state import State
from pyboatsim.constants import AXES, EPSILON
from pyboatsim.kinematics.topology import Topology

class Spring(BodyDynamicsParent):
    def __init__(
            self,
            name: str,
            body1: str,
            frame1:str,
            body2:str,
            frame2:str,
            stiffness:float
        ):
        super().__init__(name=name, body_names=[body1, body2])
        self.stiffness = stiffness
        self.body1 = body1
        self.frame1 = frame1
        self.body2 = body2
        self.frame2 = frame2
    
    def compute_dynamics(self, topology:Topology, body_name:str) -> typing.Tuple[np.matrix, np.matrix]:
        this_body, this_frame = ((self.body2, self.frame2), (self.body1, self.frame1))[body_name == self.body1]
        that_body, that_frame = ((self.body2, self.frame2), (self.body1, self.frame1))[body_name == self.body2]
        r_this = topology.get_transform(
            from_body_name="World", 
            from_frame_name="Identity", 
            to_body_name=this_body, 
            to_frame_name=this_frame)[:3,3]
        r_that = topology.get_transform(
            from_body_name="World", 
            from_frame_name="Identity", 
            to_body_name=that_body, 
            to_frame_name=that_frame)[:3,3]

        force = self.stiffness * (r_that - r_this)
        point_of_application = topology.get_transform("World", "Identity", this_body, this_frame)[:3,3]
        return [(force, point_of_application),]
