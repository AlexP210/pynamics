import os

import numpy as np
import trimesh as tri

import pynamics.kinematics.topology as topo
import pynamics.kinematics.joint as joint
import pynamics.dynamics as dyn
import pynamics.sim as sim
import pynamics.visualizer as vis

# Create the Pendulum body
# Here, a thin rod with m=1, L=1, body frame located
# at one of the ends, with cylinder axis aligned with X axis
pendulum_body = topo.Body(
    mass=1,
    center_of_mass=np.matrix([0.5, 0, 0]).T,
    inertia_matrix=np.matrix([
        [0, 0, 0],
        [0, 1/3, 0],
        [0, 0, 1/3]
    ]),
)

# Create the Topology by connecting Bodies to each other, or World
pendulum_world = topo.Topology()
pendulum_world.add_connection(
    parent_body_name="World", 
    parent_frame_name="Identity", 
    child_body=pendulum_body, 
    child_body_name="Pendulum", 
    joint=joint.RevoluteJoint(1)
)
# Initialize the joint with q=0, dq/dt = 0
pendulum_world.joints["Pendulum"].set_configuration(np.matrix([[0]]))
pendulum_world.joints["Pendulum"].set_configuration_d(np.matrix([[0]]))

# Create the sim object, with gravity, joint damping, and a DC motor
pendulum_sim = sim.Sim(
    topology=pendulum_world,
    body_dynamics={
        "gravity": dyn.Gravity(
            g=-9.81,
            direction=np.matrix([0,0,1]).T,
            body_names=["Pendulum",]
        )
    },
    joint_dynamics={
        "damp": dyn.JointDamping(
            damping_factor=0.1,
            joint_names="Pendulum"
        ),
        "motor": dyn.RevoluteDCMotor(
            joint_name="Pendulum",
            electromotive_constant=0.25,
            resistance=1,
            inductance=1,
            voltage=10
        )
    }
)

# Run the simulation
pendulum_sim.simulate(delta_t=30, dt=0.01, verbose=True)

# Plot the results
import matplotlib.pyplot as plt
time = pendulum_sim.data_history["Time"]
joint_position = pendulum_sim.data_history["Pendulum / Position 0"]
joint_velocity = pendulum_sim.data_history["Pendulum / Velocity 0"]
fig, ax = plt.subplots(1,2)
ax[0].plot(time, joint_position)
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Joint Position (rad)")
ax[1].plot(time, joint_velocity, label="Joint Velocity (rad/s)")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Joint Velocity (rad/s)")
plt.tight_layout()
plt.savefig(os.path.join("pynamics", "example", "pendulum.png"))

# Visualize the results
visualizer = vis.Visualizer(
    topology=pendulum_world,
    visualization_models={
        ("Pendulum", "Identity"): tri.load(
            file_obj=os.path.join("pynamics", "models", "common", "rod.obj"),
            file_type="obj",
            force="mesh"
        )
    },
    sim=pendulum_sim
)
visualizer.animate(
    save_path=os.path.join("pynamics", "example", "pendulum.mp4"),
    verbose=True
)
