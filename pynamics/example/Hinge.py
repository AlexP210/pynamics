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

    body = topo.Body(
        mass=100,
        center_of_mass=np.matrix([0, 0, 0]).T,
        inertia_matrix=np.matrix([
            # Point mass
            [100/12*(0.4**2+2**2),0,0],
            [0,100/12*(0.4**2+2**2),0],
            [0,0,100/12*(2**2+2**2)]
        ])
    )
    tip_frame_1 = topo.Frame(translation=np.matrix([1,0,0]).T)
    body.add_frame(tip_frame_1, "Tip Frame 1")
    tip_frame_2 = topo.Frame(translation=np.matrix([-1,0,0]).T)
    body.add_frame(tip_frame_2, "Tip Frame 2")

    topology = topo.Topology()
    jnt = joint.RevoluteJoint(1)
    # jnt = joint.FreeJoint()
    topology.add_connection("World", "Identity", body.copy(), "Body", jnt)
    
    visualizer = Visualizer(
        topology=topology,
        visualization_models={
            (f"Body", "Identity"): trimesh.load(
                file_obj=os.path.join("models", "cup", "cup_boundary_poked.obj"), 
                file_type="obj", 
                force="mesh"),
        }
    )

    sim = Sim(
        topology=topology,
        body_dynamics={
            "Torque1": ConstantBodyForce(
                force=np.matrix([0,0,10]).T,
                application_position=("Body", "Tip Frame 1"),
                application_orientation=("Body", "Identity"),
                body_names=["Body",]
            ),
            # "Torque2": ConstantBodyForce(
            #     force=np.matrix([0,0,-10]).T,
            #     application_position=("Body", "Tip Frame 2"),
            #     application_orientation=("Body", "Tip Frame 2"),
            #     body_names=["Body",]
            # ),

        }
    )

    sim.simulate(delta_t=50, dt=0.1, verbose=True)
    t = sim.data["Time"]
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
    fig, ax = plt.subplots(2,3)
    for i_j in range(6): 
        i = i_j//3
        j = i_j%3
        v = sim.data["Bodies"]["Body"][f"Acceleration {i_j}"]
        ax[i,j].plot(t, v, linewidth=1, c="k")
        ax[i,j].set_xlabel("Time")
        ax[i,j].set_ylabel(f"Acceleration {i_j}")
    fig.suptitle(f"Body")
    plt.savefig("acceleration.png")
    plt.cla()
    plt.clf()
    plt.cla()
    plt.close()

    # Body Velocities
    fig, ax = plt.subplots(2,3)
    for i_j in range(6): 
        i = i_j//3
        j = i_j%3
        v = sim.data["Bodies"]["Body"][f"Velocity {i_j}"]
        ax[i,j].plot(t, v, linewidth=1, c="k")
        ax[i,j].set_xlabel("Time")
        ax[i,j].set_ylabel(f"Velocity {i_j}")
    fig.suptitle(f"Body")
    plt.savefig("velocity.png")
    plt.clf()
    plt.cla()
    plt.close()

    visualizer.add_sim_data(sim)
    visualizer.animate(fps=60, save_path="sim.mp4")


