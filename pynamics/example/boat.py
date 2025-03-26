import numpy as np
import trimesh
import os

import matplotlib.pyplot as plt

from pynamics.dynamics import Gravity, Buoyancy, QuadraticDrag, RevoluteMotor, JointDamping
import pynamics.kinematics.topology as topo
import pynamics.kinematics.joint as joint
from pynamics.boatsim import Sim
from pynamics.visualizer import Visualizer

if __name__ == "__main__":

    for i in range(1,11):
        boat_body = topo.Body(
            mass=100,
            center_of_mass=np.matrix([0, 0, 0]).T,
            inertia_matrix=np.matrix([
                # Point mass
                [100/12*(0.4**2+2**2),0,0],
                [0,100/12*(0.4**2+2**2),0],
                [0,0,100/12*(2**2+2**2)]
            ])
        )
        water_wheel_1_frame = topo.Frame(translation=np.matrix([0,-1.25,0.2*(i/10)]).T)
        water_wheel_2_frame = topo.Frame(translation=np.matrix([0,1.25,0.2*(i/10)]).T)
        boat_body.add_frame(water_wheel_1_frame, "Water Wheel 1 Frame")
        boat_body.add_frame(water_wheel_2_frame, "Water Wheel 2 Frame")
        water_wheel_body = topo.Body(
            mass=100,
            center_of_mass=np.matrix([0, 0, 0]).T,
            inertia_matrix=np.matrix([
                # Point mass
                [100/12*(3*0.5**2+0.5**2),0,0],
                [0,100/2 * 0.5**2,0],
                [0,0,100/12*(3*0.5**2+0.5**2)]
            ])
        )

        boat = topo.Topology()
        boat.add_connection("World", "Identity", boat_body.copy(), "Boat", joint.FreeJoint())
        boat.add_connection("Boat", "Water Wheel 1 Frame", water_wheel_body.copy(), "Water Wheel 1", joint.RevoluteJoint(1))
        boat.add_connection("Boat", "Water Wheel 2 Frame", water_wheel_body.copy(), "Water Wheel 2", joint.RevoluteJoint(1))
        
        boat.joints["Boat"].set_configuration(np.matrix([0,0,0 ,0, 0, 0.1]).T)
        boat.joints["Boat"].set_configuration_d(np.matrix([0,0,0 ,0, 0, 0]).T)
        # water_world.joints["Buoy"].set_configuration(np.matrix([0, 0, 2]).T)
        water_world_vis = Visualizer(
            topology=boat,
            visualization_models={
                (f"Boat", "Identity"): trimesh.load(
                    file_obj=os.path.join("models", "cup", "cup_boundary_poked.obj"), 
                    file_type="obj", 
                    force="mesh"),
                (f"Water Wheel 1", "Identity"): trimesh.load(
                    file_obj=os.path.join("models", "cup", "water_wheel.obj"), 
                    file_type="obj", 
                    force="mesh"),
                (f"Water Wheel 2", "Identity"): trimesh.load(
                    file_obj=os.path.join("models", "cup", "water_wheel.obj"), 
                    file_type="obj", 
                    force="mesh"),
            }
        )

        water_world_sim = Sim(
            topology=boat,
            body_dynamics={
                "gravity": Gravity(-9.81, 2),
                "buoyancy": Buoyancy(
                    buoyancy_models={
                        "Boat": trimesh.load(
                            file_obj=os.path.join("models", "cup", "cup_boundary_poked.obj"), 
                            file_type="obj", 
                            force="mesh" 
                        ),
                        "Water Wheel 1": trimesh.load(
                            file_obj=os.path.join("models", "cup", "water_wheel.obj"), 
                            file_type="obj", 
                            force="mesh"),
                        "Water Wheel 2": trimesh.load(
                            file_obj=os.path.join("models", "cup", "water_wheel.obj"), 
                            file_type="obj", 
                            force="mesh"),

                    },
                ),
                "drag": QuadraticDrag(
                    drag_models={
                        "Boat": trimesh.load(
                            file_obj=os.path.join("models", "cup", "cup_boundary_poked.obj"), 
                            file_type="obj", 
                            force="mesh" ),
                        "Water Wheel 1": trimesh.load(
                            file_obj=os.path.join("models", "cup", "water_wheel.obj"), 
                            file_type="obj", 
                            force="mesh"),
                        "Water Wheel 2": trimesh.load(
                            file_obj=os.path.join("models", "cup", "water_wheel.obj"), 
                            file_type="obj", 
                            force="mesh"),
                    }
                )
            },
            joint_dynamics={
                "motor1": RevoluteMotor(
                    joint_name="Water Wheel 1",
                    electromotive_constant=100,
                    resistance=1,
                    inductance=1,
                    voltage=200),
                "motor2": RevoluteMotor(
                    joint_name="Water Wheel 2",
                    electromotive_constant=100,
                    resistance=1,
                    inductance=1,
                    voltage=200),
                # "damp": JointDamping(
                #     damping_factor=0.1,
                #     joint_names=["Water Wheel 1", "Water Wheel 2"]
                # )
            }
        )

        water_world_sim.simulate(delta_t=20, dt=0.01, verbose=True)
        water_world_sim.save_data(f"./Boat_Test_{i}.csv")
        plt.plot(
            water_world_sim.data_history["Time"], 
            water_world_sim.data_history["Boat / Velocity 3"],
            label=i)
        water_world_vis.add_sim_data(water_world_sim)
        water_world_vis.animate(
            save_path=f"Boat_Test_{i}.mp4", 
            verbose=True)
    plt.legend()
    plt.show()


