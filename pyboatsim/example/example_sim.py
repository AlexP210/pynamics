import numpy as np

from pyboatsim.dynamics import Gravity, JointDamping
import pyboatsim.kinematics.topology as topo
from pyboatsim.boatsim import Sim
from pyboatsim.visualizer import Visualizer

if __name__ == "__main__":
    # Define the bodies
    head_radius = 1
    torso_length = 2
    limb_length = 1

    torso = topo.Body(
        mass = 4,
        center_of_mass=np.matrix([torso_length/2, 0, 0]),
        inertia_matrix=np.matrix([
            [0, 0, 0],
            [0, (1/12)*4*torso_length**2, 0],
            [0, 0, (1/12)*4*torso_length**2],
        ])
    )

    limb = topo.Body(
        mass = 1,
        center_of_mass=np.matrix([limb_length/2, 0, 0]),
        inertia_matrix=np.matrix([
            [0, 0, 0],
            [0, (1/12)*1*limb_length**2, 0],
            [0, 0, (1/12)*1*limb_length**2],
        ])
    )

    head = topo.Body(
        mass = 2,
        center_of_mass=np.matrix([head_radius, 0, 0]),
        inertia_matrix=np.matrix([
            [(2/5)*2*head_radius**2, 0, 0],
            [0, (2/5)*2*head_radius**2, 0],
            [0, 0, (2/5)*2*head_radius**2],
        ])
    )

    # torso.add_frame(
    #     frame=topo.Frame(translation=)
    # )

