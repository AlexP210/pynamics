import numpy as np
import trimesh

from pyboatsim.dynamics import Gravity, RevoluteMotor, JointDamping
import pyboatsim.kinematics.topology as topo
import pyboatsim.kinematics.joint as joint
from pyboatsim.boatsim import Sim
from pyboatsim.visualizer import Visualizer

if __name__ == "__main__":

    pendulum_body = topo.Body(
        mass=0.1,
        center_of_mass=np.matrix([1, 0, 0]).T,
        inertia_matrix=np.matrix([
            # Point mass
            [0,0,0],
            [0,0,0],
            [0,0,0]
        ])
    )
    end = topo.Frame(
        translation=np.matrix([1.0,0.0,0.0]).T, 
    )
    pendulum_body.add_frame(end, "Arm End")

    pendulum_world = topo.Topology()
    pendulum_world.add_connection("World", "Identity", pendulum_body.copy(), "Pendulum Body", joint.RevoluteJoint(1))
    pendulum_world.joints["Pendulum Body"].set_configuration(np.matrix([[np.pi/2]]).T)
    pendulum_world_vis = Visualizer(
        topology=pendulum_world,
        visualization_models={
            (f"Pendulum Body", "Identity"): trimesh.load(
                file_obj="/home/alex/Projects/PyBoAtSim/models/link/Link1m.obj", 
                file_type="obj", 
                force="mesh"),
        }
    )

    pendulum_world_sim = Sim(
        topology=pendulum_world,
        body_dynamics=[
            Gravity("gravity", -9.81, 2)
        ],
        joint_dynamics=[
            RevoluteMotor(
                name="motor",
                joint_name="Pendulum Body",
                electromotive_constant=0.05,
                resistance=1,
                inductance=0.5,
                voltage=10,
            ),
            JointDamping(
                name="damp",
                joint_names=["Pendulum Body",],
                damping_factor=0.1
            )
        ]
    )

    pendulum_world_sim.simulate(delta_t=15, dt=0.01, verbose=True)
    pendulum_world_vis.add_sim_data(pendulum_world_sim)
    pendulum_world_vis.animate(save_path="Motor_Test.mp4", verbose=True)

