import numpy as np
import trimesh
import os

import matplotlib.pyplot as plt

from pynamics.dynamics import Gravity, Buoyancy, QuadraticDrag, RevoluteDCMotor, JointDamping, ConstantJointForce, ConstantBodyForce
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
    water_wheel_1_frame = topo.Frame(translation=np.matrix([0,-1.25,0]).T)
    water_wheel_2_frame = topo.Frame(translation=np.matrix([0,1.25,0]).T)
    water_wheel_tip_frame_1 = topo.Frame(translation=np.matrix([1,0,0]).T)
    water_wheel_tip_frame_2 = topo.Frame(translation=np.matrix([-1,0,0]).T)
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
    water_wheel_body.add_frame(water_wheel_tip_frame_1, "Tip Frame 1")
    water_wheel_body.add_frame(water_wheel_tip_frame_2, "Tip Frame 2")

    boat = topo.Topology()
    boat.add_connection("World", "Identity", boat_body.copy(), "Boat", joint.FreeJoint())
    boat.add_connection("Boat", "Water Wheel 1 Frame", water_wheel_body.copy(), "Water Wheel 1", joint.RevoluteJoint(1))
    boat.add_connection("Boat", "Water Wheel 2 Frame", water_wheel_body.copy(), "Water Wheel 2", joint.RevoluteJoint(1))
    
    boat.joints["Boat"].set_configuration(np.matrix([np.cos(np.pi/2),0,np.sin(np.pi/2),0, 0,0,0.1]).T)
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
            "Torque 1 Wheel 1": ConstantBodyForce(
                force=np.matrix([0,0,1]).T,
                application_position=("Water Wheel 1", "Tip Frame 1"),
                application_orientation=("Water Wheel 1", "Tip Frame 1"),
                body_names=["Water Wheel 1",]
            ),
            # "Torque 2 Wheel 1": ConstantBodyForce(
            #     force=np.matrix([0,0,-1]).T,
            #     application_position=("Water Wheel 1", "Tip Frame 2"),
            #     application_orientation=("Water Wheel 1", "Identity"),
            #     body_names=["Water Wheel 1",]
            # ),
            "Torque 1 Wheel 2": ConstantBodyForce(
                force=np.matrix([0,0,1]).T,
                application_position=("Water Wheel 2", "Tip Frame 1"),
                application_orientation=("Water Wheel 2", "Tip Frame 1"),
                body_names=["Water Wheel 2",]
            ),
            # "Gravity" : Gravity(g=-9.81, body_names=["Boat"])
            # "Torque 2 Wheel 2": ConstantBodyForce(
            #     force=np.matrix([0,0,-1]).T,
            #     application_position=("Water Wheel 2", "Tip Frame 2"),
            #     application_orientation=("Water Wheel 2", "Identity"),
            #     body_names=["Water Wheel 2",]
            # ),
            # "gravity": Gravity(-9.81),
            # "buoyancy": Buoyancy(
            #     buoyancy_models={
            #         "Boat": trimesh.load(
            #             file_obj=os.path.join("models", "cup", "cup_boundary_poked.obj"), 
            #             file_type="obj", 
            #             force="mesh" 
            #         ),
            #         "Water Wheel 1": trimesh.load(
            #             file_obj=os.path.join("models", "cup", "water_wheel.obj"), 
            #             file_type="obj", 
            #             force="mesh"),
            #         "Water Wheel 2": trimesh.load(
            #             file_obj=os.path.join("models", "cup", "water_wheel.obj"), 
            #             file_type="obj", 
            #             force="mesh"),
            #     },
            # ),
            # "drag": QuadraticDrag(
            #     drag_models={
            #         "Boat": trimesh.load(
            #             file_obj=os.path.join("models", "cup", "cup_boundary_poked.obj"), 
            #             file_type="obj", 
            #             force="mesh" ),
            #         "Water Wheel 1": trimesh.load(
            #             file_obj=os.path.join("models", "cup", "water_wheel.obj"), 
            #             file_type="obj", 
            #             force="mesh"),
            #         "Water Wheel 2": trimesh.load(
            #             file_obj=os.path.join("models", "cup", "water_wheel.obj"), 
            #             file_type="obj", 
            #             force="mesh"),
                # },
                # drag_coefficient=0.5
            # )
        },
        joint_dynamics={
            # "motor1": RevoluteDCMotor(
            #     joint_name="Water Wheel 1",
            #     electromotive_constant=10,
            #     resistance=0.1,
            #     inductance=10,
            #     voltage=10),
            # "motor2": RevoluteDCMotor(
            #     joint_name="Water Wheel 2",
            #     electromotive_constant=10,
            #     resistance=0.1,
            #     inductance=10,
            #     voltage=10),
            # "damp": JointDamping(
            #     damping_factor=1,
            #     joint_names=["Water Wheel 1", "Water Wheel 2"]
            # ),
            # "motor1" : ConstantJointForce(
            #     force=np.matrix([2,]).T,
            #     joint_names=["Water Wheel 1",]
            # ),
            # "motor2" : ConstantJointForce(
            #     force=np.matrix([2,]).T,
            #     joint_names=["Water Wheel 2",]
            # )
        }
    )

    water_world_sim.simulate(delta_t=50, dt=0.1, verbose=True)
    # water_world_sim.joint_dynamics["motor1"].voltage = -10
    # water_world_sim.simulate(delta_t=3, dt=0.01, verbose=True)
    # water_world_sim.joint_dynamics["motor1"].voltage = 10
    # water_world_sim.simulate(delta_t=20, dt=0.01, verbose=True)

    t = water_world_sim.data["Time"]
    # # Joint Velocities
    # for body_idx, body_name in enumerate(boat.get_ordered_body_list()[1:]):
    #     jnt = boat.joints[body_name]
    #     fig, ax = plt.subplots(1,jnt.get_configuration_d().size)
    #     for i in range(jnt.get_configuration_d().size): 
    #         v = water_world_sim.data["Joints"][body_name][f"Velocity {i}"]
    #         axes = ax[i] if jnt.get_configuration_d().size > 1 else ax
    #         axes.plot(t, v, linewidth=1, c="k")
    #         axes.set_xlabel("Time")
    #         axes.set_ylabel(f"Acceleration {i}")
    #     fig.suptitle(f"{body_name} Body")
    #     plt.show()

    # Body Accelerations
    for body_idx, body_name in enumerate(boat.get_ordered_body_list()[1:]):
        fig, ax = plt.subplots(2,3)
        for i_j in range(6): 
            i = i_j//3
            j = i_j%3
            v = water_world_sim.data["Bodies"][body_name][f"Acceleration {i_j}"]
            ax[i,j].plot(t, v, linewidth=1, c="k")
            ax[i,j].set_xlabel("Time")
            ax[i,j].set_ylabel(f"Acceleration {i_j}")
        fig.suptitle(f"{body_name} Body")
        plt.savefig(f"acceleration_{body_name}.png")
        plt.cla()
        plt.clf()
        plt.cla()
        plt.close()

    # Body Velocities
    for body_idx, body_name in enumerate(boat.get_ordered_body_list()[1:]):
        fig, ax = plt.subplots(2,3)
        for i_j in range(6): 
            i = i_j//3
            j = i_j%3
            v = water_world_sim.data["Bodies"][body_name][f"Velocity {i_j}"]
            ax[i,j].plot(t, v, linewidth=1, c="k")
            ax[i,j].set_xlabel("Time")
            ax[i,j].set_ylabel(f"Velocity {i_j}")
        fig.suptitle(f"{body_name} Body")
        plt.savefig(f"velocity_{body_name}.png")
        plt.cla()
        plt.clf()
        plt.cla()
        plt.close()


    # # Motor Current
    # fig, ax = plt.subplots(1, 2)
    # for i in range(2): 
    #     I = water_world_sim.data["Joint Forces"][f"motor{i+1}"][f"Water Wheel {i+1}"][f"Current"]
    #     ax[i].plot(t, I, linewidth=1, c="k")
    #     ax[i].set_xlabel("Time")
    #     ax[i].set_ylabel(f"Motor {i+1} Current")
    # fig.suptitle(f"Motor Currents")
    # plt.show()


    # # Drag Force
    # for body_idx, body_name in enumerate(boat.get_ordered_body_list()[1:2]):
    #     fig, ax = plt.subplots(2,3)
    #     for i, f_m in enumerate(["Force", "Moment"]): 
    #         for j in range(3):
    #             v = water_world_sim.data["Body Forces"]["drag"][body_name][f"{f_m} {j}"]
    #             ax[i,j].plot(t, v, linewidth=1, c="k")
    #             ax[i,j].set_xlabel("Time")
    #             ax[i,j].set_ylabel(f"{f_m} {j}")
    #     fig.suptitle(f"{body_name} Body")
    #     plt.show()

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

    water_world_vis.add_sim_data(water_world_sim)
    water_world_vis.animate(fps=60, save_path="boat_sim.mp4")


