import os

import numpy as np
import trimesh

from pyboatsim.dynamics import Gravity, Buoyancy, QuadraticDrag
import pyboatsim.kinematics.topology as topo
import pyboatsim.kinematics.joint as joint
from pyboatsim.boatsim import Sim
from pyboatsim.visualizer import Visualizer

if __name__ == "__main__":

    body = topo.Body(
        mass=500,
        center_of_mass=np.matrix([0, 0, 0]).T,
        inertia_matrix=np.matrix([
            # Point mass
            [100,0,0],
            [0,100,0],
            [0,0,100]
        ])
    )

    water_world = topo.Topology()
    water_world.add_connection("World", "Identity", body.copy(), "Block", joint.FreeJoint())
    water_world.joints["Block"].set_configuration_d(np.matrix([0, 0, 0, 0, 0, 0]).T)
    water_world.joints["Block"].set_configuration(np.matrix([0, 0, 0, 1, 0, 0]).T)
    water_world_vis = Visualizer(
        topology=water_world,
        visualization_models={
            (f"Block", "Identity"): trimesh.load(
                file_obj=os.path.join("models", "cup", "cup_boundary_poked.obj"), 
                file_type="obj", 
                force="mesh"),
        }
    )

    water_world_sim = Sim(
        topology=water_world,
        body_dynamics={Gravity(-9.81, 2, body_names=["Block"])},
    )

    water_world_sim.simulate(delta_t=10, dt=0.01, verbose=True)
    water_world_sim.save_data("Block_Test.csv")
    water_world_vis.add_sim_data(water_world_sim)
    water_world_vis.animate(
        save_path="Block_Test.mp4", 
        verbose=True)

