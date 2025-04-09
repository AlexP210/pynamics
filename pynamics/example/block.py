import os

import numpy as np
import trimesh

from pynamics.dynamics import Gravity, Buoyancy, QuadraticDrag, ConstantBodyForce
import pynamics.kinematics.topology as topo
import pynamics.kinematics.joint as joint
from pynamics.sim import Sim
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
        body_dynamics={
            "Constant": ConstantBodyForce(
                force = np.matrix([0,0,-500]).T,
                application_position=("Block", "Identity"),
                application_orientation=("World", "Identity")
            ),
            # "Gravity": Gravity(g=-9.81),
            "Drag": QuadraticDrag(
                drag_models={
                    "Block": trimesh.load(
                        file_obj=os.path.join(HOME, "models", "common", "Cube.obj"), 
                        file_type="obj", 
                        force="mesh")
                }
            )
        },
    )

    water_world_sim.simulate(delta_t=10, dt=0.001, verbose=True)
    # water_world_sim.save_data("Block_Test.csv")
    water_world_vis.add_sim_data(water_world_sim)
    water_world_vis.animate(
        save_path="Block_Test.mp4", 
        verbose=True)

    # Body Velocities
    import matplotlib.pyplot as plt
    t = water_world_sim.data["Time"]
    for body_idx, body_name in enumerate(water_world.get_ordered_body_list()[1:2]):
        fig, ax = plt.subplots(2,3)
        for i_j in range(6): 
            i = i_j//3
            j = i_j%3
            v = water_world_sim.data["Bodies"][body_name][f"Velocity {i_j}"]
            ax[i,j].plot(t, v, linewidth=1, c="k")
            ax[i,j].set_xlabel("Time")
            ax[i,j].set_ylabel(f"Velocity {i_j}")
        fig.suptitle(f"{body_name} Body")
        plt.show()
