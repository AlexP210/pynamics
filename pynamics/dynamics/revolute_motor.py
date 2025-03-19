import typing

import numpy as np
import trimesh

from pynamics.dynamics import JointDynamicsParent
from pynamics.state import State
from pynamics.constants import AXES, EPSILON
from pynamics.kinematics.topology import Topology

from pynamics.math.linalg import R3_cross_product_matrix

class RevoluteMotor(JointDynamicsParent):
    def __init__(
            self,
            joint_name:str,
            electromotive_constant:float,
            resistance:float,
            inductance:float,
            voltage:float,
            initial_current:float = 0,
        ):
        super().__init__(joint_names=[joint_name,])
        self.electromotive_constant = electromotive_constant
        self.resistance = resistance
        self.inductance = inductance
        self.voltage = voltage
        self.current = initial_current
        self.emf = 0
    
    def compute_dynamics(self, topology:Topology, joint_name:str) -> typing.Tuple[np.matrix, np.matrix]:
        joint = topology.joints[joint_name]
        S = joint.get_motion_subspace()
        if not (sum(S[:3,0]) == 1 and sum(S[3:,0]) == 0):
            raise ValueError(f"RevoluteMotor \"{self.name}\" cannot operate on non-revolute joint \"{self.joint_name}\".")
        return [self.electromotive_constant * self.current*np.matrix(np.ones(shape=joint.get_configuration().shape)),]
    
    def update(self, topology:Topology, dt:float):
        joint = topology.joints[self.joint_names[0]]
        self.emf = self.electromotive_constant*joint.get_configuration_d()[0,0]
        di_dt = ((self.voltage - self.emf) - self.resistance * self.current) / self.inductance
        self.current += di_dt*dt

    def get_data(self):
        return {
            "Current": self.current,
            "EMF": self.emf,
            "Voltage": self.voltage,
        }