import os

import numpy as np
import trimesh

from pynamics.dynamics import Gravity, Buoyancy, QuadraticDrag
import pynamics.kinematics.topology as topo
import pynamics.kinematics.joint as joint
from pynamics.boatsim import Sim
from pynamics.visualizer import Visualizer
from pynamics.constants import HOME

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
    water_world.joints["Block"].set_configuration(np.matrix([1/np.sqrt(4),1/np.sqrt(4),1/np.sqrt(4),1/np.sqrt(4), 1, 1, 1]).T)
    water_world_vis = Visualizer(
        topology=water_world,
        visualization_models={
            (f"Block", "Identity"): trimesh.load(
                file_obj=os.path.join(HOME, "models", "common", "Cube.obj"), 
                file_type="obj", 
                force="mesh"),
        }
    )

    water_world_sim = Sim(
        topology=water_world,
        body_dynamics={"gravity": Gravity(-9.81, 2, body_names=["Block"])},
    )

    water_world_sim.simulate(delta_t=5, dt=0.01, verbose=True)
    water_world_sim.save_data("Block_Test.csv")
    water_world_vis.add_sim_data(water_world_sim)
    water_world_vis.animate(
        save_path="Block_Test.mp4", 
        verbose=True)

