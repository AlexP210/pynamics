import numpy as np
import trimesh

from pyboatsim.dynamics import Gravity, JointDamping, Spring, RevoluteMotor
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

    bouncy_body = topo.Body(mass=1)

    spring_world = topo.Topology()
    spring_world.add_connection("World", "Identity", pendulum_body.copy(), "Pendulum Body", joint.RevoluteJoint(1))
    spring_world.add_connection("World", "Identity", bouncy_body.copy(), "Bouncy Body", joint.FreeJoint())
    spring_world.joints["Bouncy Body"].set_configuration(np.matrix([0,0,0,1,0,0]).T)
    spring_world_vis = Visualizer(
        topology=spring_world,
        visualization_models={
            (f"Pendulum Body", "Identity"): trimesh.load(
                file_obj="/home/alex/Projects/PyBoAtSim/models/link/Link1m.obj", 
                file_type="obj", 
                force="mesh"),
            (f"Bouncy Body", "Identity"): trimesh.load(
                file_obj="/home/alex/Projects/PyBoAtSim/models/common/UnitBall.obj", 
                file_type="obj", 
                force="mesh"),
        }
    )

    spring_world_sim = Sim(
        topology=spring_world,
        body_dynamics=[
            Spring(
                name="spring",
                body1="Pendulum Body",
                frame1="Arm End",
                body2="Bouncy Body",
                frame2="Identity",
                stiffness=10
            ),
            Gravity("gravity", -9.81, 2)
        ]
    )

    spring_world_sim.simulate(delta_t=10, dt=0.01, verbose=True)
    spring_world_vis.add_sim_data(spring_world_sim)
    spring_world_vis.animate(save_path="Spring_Test.mp4", verbose=True)

