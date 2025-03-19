import numpy as np
import trimesh

from pyboatsim.dynamics import Gravity, JointDamping, Spring
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

    net_world = topo.Topology()
    N = 9
    s = 1
    springs = {}
    for i in range(N):
        for j in range(N):
            net_world.add_frame(
                body_name="World", 
                frame=topo.Frame(translation=np.matrix([i*s, 0, j*s]).T),
                frame_name=f"{i}{j} Frame"
            )
            if i%(N-1) and j%(N-1): joint_to_use = joint.TranslationJoint()
            else: joint_to_use = joint.FixedJoint()
            net_world.add_connection(
                parent_body_name="World",
                parent_frame_name=f"{i}{j} Frame",
                child_body=bouncy_body.copy(),
                child_body_name=f"{i}{j} Body",
                joint=joint_to_use
            )
            if i*j: 
                springs[f"{i-1}{j}-{i}{j} Spring"] = Spring(
                        body1=f"{i-1}{j} Body", frame1="Identity",
                        body2=f"{i}{j} Body", frame2=f"Identity",
                        stiffness=50
                    )
                springs[f"{i}{j-1}-{i}{j} Spring"] = Spring(
                        body1=f"{i}{j-1} Body", frame1="Identity",
                        body2=f"{i}{j} Body", frame2=f"Identity",
                        stiffness=50
                    ),
    net_world.joints[f"{N//2}{N//2} Body"].set_configuration(np.matrix([0, -1, 0]).T)

    net_world_vis = Visualizer(
        topology=net_world,
        visualization_models={
            (f"{i}{j} Body", "Identity"): trimesh.load(
                file_obj="/home/alex/Projects/PyBoAtSim/models/common/UnitBall.obj", 
                file_type="obj", 
                force="mesh")
            for i in range(N)
            for j in range(N)
        }
    )

    net_world_sim = Sim(
        topology=net_world, 
        body_dynamics=springs, 
        joint_dynamics={"damping": JointDamping(2)}
    )

    net_world_sim.simulate(5, 0.02, verbose=True)
    net_world_vis.add_sim_data(net_world_sim)
    net_world_vis.animate("Net_Test2.mp4", verbose=True)
