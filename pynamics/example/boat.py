import numpy as np
import trimesh
import os

import matplotlib.pyplot as plt

from pynamics.dynamics import Gravity, Buoyancy, QuadraticDrag, RevoluteDCMotor, JointDamping, ConstantJointForce
import pynamics.kinematics.topology as topo
import pynamics.kinematics.joint as joint
from pynamics.sim import Sim
from pynamics.visualizer import Visualizer

if __name__ == "__main__":

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
    water_wheel_1_frame = topo.Frame(translation=np.matrix([0,-1.25,0.1]).T)
    water_wheel_2_frame = topo.Frame(translation=np.matrix([0,1.25,0.1]).T)
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
    
    boat.joints["Boat"].set_configuration(np.matrix([1,0,0,0, 0,0,0.1]).T)
    # boat.joints["Boat"].set_configuration(np.matrix([0,0,0.1]).T)

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
            "gravity": Gravity(-9.81),
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
                },
                # drag_coefficient=0.5
            )
        },
        joint_dynamics={
            "motor1": RevoluteDCMotor(
                joint_name="Water Wheel 1",
                electromotive_constant=10,
                resistance=0.1,
                inductance=10,
                voltage=10),
            "motor2": RevoluteDCMotor(
                joint_name="Water Wheel 2",
                electromotive_constant=10,
                resistance=0.1,
                inductance=10,
                voltage=10),
            "damp": JointDamping(
                damping_factor=0.1,
                joint_names=["Water Wheel 1", "Water Wheel 2"]
            ),
            # "motor1" : ConstantJointForce(
            #     force=np.matrix([10,]).T,
            #     joint_names=["Water Wheel 1",]
            # ),
            # "motor2" : ConstantJointForce(
            #     force=np.matrix([10,]).T,
            #     joint_names=["Water Wheel 2",]
            # )
        }
    )

    water_world_sim.simulate(delta_t=5.195, dt=0.005, verbose=True)
    water_world_vis.add_sim_data(water_world_sim)
    water_world_vis.animate(fps=60, save_path="boat_sim.mp4")
    t = water_world_sim.data["Time"]

    # # Joint Accelerations
    # for body_idx, body_name in enumerate(boat.get_ordered_body_list()[1:]):
    #     jnt = boat.joints[body_name]
    #     fig, ax = plt.subplots(1,jnt.get_configuration_d().size)
    #     for i in range(jnt.get_configuration_d().size): 
    #         v = water_world_sim.data["Joints"][body_name][f"Acceleration {i}"]
    #         ax[i].plot(t, v, linewidth=5, c="k")
    #         ax[i].set_xlabel("Time")
    #         ax[i].set_ylabel(f"Acceleration {i}")
    #     fig.suptitle(f"{body_name} Body")
    #     plt.show()

    # Body Velocities
    for body_idx, body_name in enumerate(boat.get_ordered_body_list()[1:2]):
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

    # Drag Force
    for body_idx, body_name in enumerate(boat.get_ordered_body_list()[1:2]):
        fig, ax = plt.subplots(2,3)
        for i, f_m in enumerate(["Force", "Moment"]): 
            for j in range(3):
                v = water_world_sim.data["Body Forces"]["drag"][body_name][f"{f_m} {j}"]
                ax[i,j].plot(t, v, linewidth=1, c="k")
                ax[i,j].set_xlabel("Time")
                ax[i,j].set_ylabel(f"{f_m} {j}")
        fig.suptitle(f"{body_name} Body")
        plt.show()

    # fig, ax = plt.subplots(1,2)
    # for i in range(1, 3):
    #     v = water_world_sim.data["Joint Forces"][f"motor{i}"][f"Water Wheel {i}"]["Current"]
    #     ax[i-1].plot(t, v, linewidth=5, c="k")
    #     ax[i-1].set_xlabel("Time")
    #     ax[i-1].set_ylabel(f"Acceleration {i_j}")
    # fig.suptitle(f"{body_name} Body")
    # plt.show()


    # fig, ax = plt.subplots(1,3)
    # for body_idx, body_name in enumerate(boat.get_ordered_body_list()[1:]):
    #     for i in range(3):
    #         f = water_world_sim.data["Body Forces"]["drag"][body_name][f"Moment {i}"]
    #         ax[body_idx].scatter(t, f, label=f"{body_name} Moment {i}")
    # plt.title("Drag Moment Components")
    # plt.legend()
    # plt.show()

    # water_world_vis.animate(fps=1000, save_path="boat_sim.mp4")


