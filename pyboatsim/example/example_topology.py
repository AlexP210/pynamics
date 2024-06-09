import numpy as np
import trimesh
import tqdm

from pyboatsim.kinematics.topology import Body, Frame, Topology
from pyboatsim.kinematics.joint import RevoluteJoint, FreeJoint, FixedJoint
from pyboatsim.visualizer import Visualizer

body = Body(
    mass=1,
    center_of_mass=np.matrix([0,0.0,0.0]).T,
    inertia_matrix=np.matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
)
base_mounting_point = Frame(
    translation=np.matrix([2.0,0.0,0.0]).T, 
    rotation=Frame.get_rotation_matrix(0.0, np.matrix([0.0,0.0,0.0]).T)
)
short_end = Frame(
    translation=np.matrix([0.25,0.1,0.0]).T, 
    rotation=Frame.get_rotation_matrix(0.0, np.matrix([0.0,0.0,0.0]).T)
)
long_end = Frame(
    translation=np.matrix([1.0,0.1,0.0]).T, 
    rotation=Frame.get_rotation_matrix(0.0, np.matrix([0.0,0.0,0.0]).T)
)
long_end_for_yaw = Frame(
    translation=np.matrix([1.0,0.0,0.1]).T, 
    rotation=Frame.get_rotation_matrix(0.0, np.matrix([0.0,0.0,0.0]).T)
)
modules_end1 = Frame(
    translation=np.matrix([0.0,0.0,-3.0]).T, 
    rotation=Frame.get_rotation_matrix(0.0, np.matrix([0.0,0.0,0.0]).T)
)
modules_end2 = Frame(
    translation=np.matrix([0.0,0.0,3.0]).T, 
    rotation=Frame.get_rotation_matrix(0.0, np.matrix([0.0,0.0,0.0]).T)
)

module_1_location = Frame(
    translation=np.matrix([4.0,3.0,0.0]).T, 
    rotation=Frame.get_rotation_matrix(0.0, np.matrix([0.0,0.0,0.0]).T)
)
module_2_location = Frame(
    translation=np.matrix([4.0,0.0,3.0]).T, 
    rotation=Frame.get_rotation_matrix(np.pi/2, np.matrix([1.0,0.0,0.0]).T)
)


base = body.copy()
module1 = body.copy()
module2 = body.copy()
roll_body = body.copy()
pitch_body_1 = body.copy()
pitch_body_2 = body.copy()
yaw_body = body.copy()


base.add_frame(base_mounting_point, "Base to Roll Body")
base.add_frame(module_1_location, "Base to Module 1")
base.add_frame(module_2_location, "Base to Module 2")
base.add_frame(modules_end1, "End 1")
base.add_frame(modules_end2, "End 2")
module1.add_frame(modules_end1, "End 1")
module1.add_frame(modules_end2, "End 2")
module2.add_frame(modules_end1, "End 1")
module2.add_frame(modules_end2, "End 2")

roll_body.add_frame(short_end, "Roll Body to Pitch Body 1")
pitch_body_1.add_frame(long_end, "Pitch Body 1 to Pitch Body 2")
pitch_body_2.add_frame(long_end_for_yaw, "Pitch Body 2 to Yaw Body")
yaw_body.add_frame(short_end, "End Effector")


robot = Topology()
robot.add_connection("World", "Identity", base, "Base Body", joint=FixedJoint())
robot.add_connection(
    "Base Body", "Base to Module 1", module1, "Module 1", joint=FixedJoint())
robot.add_connection(
    "Base Body", "Base to Module 2", module2, "Module 2", joint=FixedJoint())

robot.add_connection(
    "Base Body", "Base to Roll Body", roll_body, "Roll Body",
    joint=RevoluteJoint(0))
robot.add_connection(
    "Roll Body", "Roll Body to Pitch Body 1", pitch_body_1, "Pitch Body 1",
    joint=RevoluteJoint(1))
robot.add_connection(
    "Pitch Body 1", "Pitch Body 1 to Pitch Body 2", pitch_body_2, "Pitch Body 2",
    joint=RevoluteJoint(1))
robot.add_connection(
    "Pitch Body 2", "Pitch Body 2 to Yaw Body", yaw_body, "Yaw Body",
    joint=RevoluteJoint(2))

robot_vis = Visualizer(
    topology=robot,
    visualization_models={
        ("Base Body", "Identity"): trimesh.load(
            file_obj="/home/alex/Projects/PyBoAtSim/models/link/Base.obj", 
            file_type="obj", 
            force="mesh"),
        ("Module 1", "Identity"): trimesh.load(
            file_obj="/home/alex/Projects/PyBoAtSim/models/link/Base.obj", 
            file_type="obj", 
            force="mesh"),
        ("Module 2", "Identity"): trimesh.load(
            file_obj="/home/alex/Projects/PyBoAtSim/models/link/Base.obj", 
            file_type="obj", 
            force="mesh"),
        ("Roll Body", "Identity"): trimesh.load(
            file_obj="/home/alex/Projects/PyBoAtSim/models/link/Link0p25m.obj", 
            file_type="obj", 
            force="mesh"),
        ("Pitch Body 1", "Identity"): trimesh.load(
            file_obj="/home/alex/Projects/PyBoAtSim/models/link/Link1m.obj", 
            file_type="obj", 
            force="mesh"),
        ("Pitch Body 2", "Identity"): trimesh.load(
            file_obj="/home/alex/Projects/PyBoAtSim/models/link/Link1m.obj", 
            file_type="obj", 
            force="mesh"),
        ("Yaw Body", "Identity"): trimesh.load(
            file_obj="/home/alex/Projects/PyBoAtSim/models/link/Link0p25m.obj", 
            file_type="obj", 
            force="mesh"),
    })

robot.joints["Roll Body"].set_configuration(np.matrix([[np.pi/4]]))
robot.joints["Pitch Body 1"].set_configuration(np.matrix([[np.pi/4]]))
robot.joints["Pitch Body 2"].set_configuration(np.matrix([[np.pi/4]]))
robot.joints["Yaw Body"].set_configuration(np.matrix([[np.pi/4]]))

double_pendulum = Topology()

arm = body.copy()
arm_end = Frame(
    translation=np.matrix([1.0,0.1,0.0]).T, 
    rotation=Frame.get_rotation_matrix(0.0, np.matrix([0.0,0.0,0.0]).T)
)
arm.add_frame(arm_end, "Arm End")
double_pendulum.add_connection("World", "Identity", arm, "Arm 1", RevoluteJoint(2))
double_pendulum.add_connection("Arm 1", "Arm End", arm, "Arm 2", RevoluteJoint(2))

double_pendulum_vis = Visualizer(
    topology=double_pendulum,
    visualization_models={
        ("Arm 1", "Identity"): trimesh.load(
            file_obj="/home/alex/Projects/PyBoAtSim/models/link/Link1m.obj", 
            file_type="obj", 
            force="mesh"),
        ("Arm 2", "Identity"): trimesh.load(
            file_obj="/home/alex/Projects/PyBoAtSim/models/link/Link1m.obj", 
            file_type="obj", 
            force="mesh"),
    }
)


