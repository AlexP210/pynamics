import numpy as np
import trimesh
import tqdm

from pyboatsim.kinematics.topology import Body, Frame, Articulation, Topology
from pyboatsim.visualizer import Visualizer

body = Body(
    mass=1,
    center_of_mass=np.matrix([0,0.0,0.0]).T,
    inertia_matrix=np.matrix([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
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
robot.add_connection("World", "Identity", base, "Base Body")
robot.add_connection(
    "Base Body", "Base to Module 1", module1, "Module 1")
robot.add_connection(
    "Base Body", "Base to Module 2", module2, "Module 2")

robot.add_connection(
    "Base Body", "Base to Roll Body", roll_body, "Roll Body",
    constraints=Articulation.ROTATE_X)
robot.add_connection(
    "Roll Body", "Roll Body to Pitch Body 1", pitch_body_1, "Pitch Body 1",
    constraints=Articulation.ROTATE_Y)
robot.add_connection(
    "Pitch Body 1", "Pitch Body 1 to Pitch Body 2", pitch_body_2, "Pitch Body 2",
    constraints=Articulation.ROTATE_Y)
robot.add_connection(
    "Pitch Body 2", "Pitch Body 2 to Yaw Body", yaw_body, "Yaw Body",
    constraints=Articulation.ROTATE_Z)

vis = Visualizer(
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
vis.view()

links = (
    ("Roll Body", "Identity", "Roll Body to Pitch Body 1"), 
    ("Pitch Body 1", "Identity", "Pitch Body 1 to Pitch Body 2"), 
    ("Pitch Body 2", "Identity","Pitch Body 2 to Yaw Body"), 
    ("Yaw Body", "Identity", 'End Effector')
)
environment = (
    ("Base Body", "End 1", "End 2"),
    ("Module 1", "End 1", "End 2"),
    ("Module 2", "End 1", "End 2"),

)
def closestDistanceBetweenLines(a0,a1,b0,b1,clampAll=False,clampA0=False,clampA1=False,clampB0=False,clampB1=False):

    ''' Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
        Return the closest points on each segment and their distance
    '''

    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0=True
        clampA1=True
        clampB0=True
        clampB1=True


    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)
    
    _A = A / magA
    _B = B / magB
    
    cross = np.cross(_A, _B)
    denom = np.linalg.norm(cross)**2
    
    
    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A,(b0-a0))
        
        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(_A,(b1-a0))
            
            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if np.absolute(d0) < np.absolute(d1):
                        return a0,b0,np.linalg.norm(a0-b0)
                    return a0,b1,np.linalg.norm(a0-b1)
                
                
            # Is segment B after A?
            elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1,b0,np.linalg.norm(a1-b0)
                    return a1,b1,np.linalg.norm(a1-b1)
                
                
        # Segments overlap, return distance between parallel segments
        return None,None,np.linalg.norm(((d0*_A)+a0)-b0)
        
    
    
    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0)
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA/denom
    t1 = detB/denom

    pA = a0 + (_A * t0) # Projected closest point on segment A
    pB = b0 + (_B * t1) # Projected closest point on segment B


    # Clamp projections
    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > magA:
            pA = a1
        
        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > magB:
            pB = b1
            
        # Clamp projection A
        if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
            dot = np.dot(_B,(pA-b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)
    
        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
            dot = np.dot(_A,(pB-a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    
    return pA,pB,np.linalg.norm(pA-pB)        

data = {
    }
for theta_1 in tqdm.tqdm(np.linspace(-np.pi, np.pi, 20)):
    for theta_2 in np.linspace(-np.pi, np.pi, 20):
        for theta_3 in np.linspace(-np.pi, np.pi, 20):
            for theta_4 in np.linspace(-np.pi, np.pi, 20):
                theta = (theta_1, theta_2, theta_3, theta_4)
                for i in range(4):
                    label = f"Theta{i+1}"
                    if label in data: data[label].append(theta[i])
                    else: data[label] = [theta[i],]
                robot.set_articulation("Roll Body", np.array([0,0,0,theta_1,0,0]))
                robot.set_articulation("Pitch Body 1", np.array([0,0,0,0,theta_2,0]))
                robot.set_articulation("Pitch Body 2", np.array([0,0,0,0,theta_3,0]))
                robot.set_articulation("Yaw Body", np.array([0,0,0,0,0,theta_4]))
                distances = []
                for link_1_idx in range(1,len(links)):
                    link_1_name, link_1_in_frame, link_1_out_frame = links[link_1_idx]
                    link_1_in = np.array(robot.get_transform("World", "Identity", link_1_name, link_1_in_frame))[:3,3]
                    link_1_out = np.array(robot.get_transform("World", "Identity", link_1_name, link_1_out_frame))[:3,3].T
                    for link_2_idx in range(link_1_idx-1):
                        link_2_name, link_2_in_frame, link_2_out_frame = links[link_2_idx]
                        link_2_in = np.array(robot.get_transform("World", "Identity", link_2_name, link_2_in_frame))[:3,3].T
                        link_2_out = np.array(robot.get_transform("World", "Identity", link_2_name, link_2_out_frame))[:3,3].T
                        _, _, dist = closestDistanceBetweenLines(
                            link_1_in, link_1_out, link_2_in, link_2_out, clampAll=True
                        )
                        label = f"{link_1_name} to {link_2_name}"
                        if label in data: data[label].append(dist)
                        else: data[label] = [dist,]
                    for env_idx in range(len(environment)):
                        env_name, env_in_frame, env_out_frame = environment[env_idx]
                        env_in = np.array(robot.get_transform("World", "Identity", env_name, env_in_frame))[:3,3].T
                        env_out = np.array(robot.get_transform("World", "Identity", env_name, env_out_frame))[:3,3].T
                        _, _, dist = closestDistanceBetweenLines(
                            link_1_in, link_1_out, env_in, env_out, clampAll=True
                        )
                        label = f"{link_1_name} to {env_name}"
                        if label in data: data[label].append(dist)
                        else: data[label] = [dist,]

                
import pandas as pd
pd.DataFrame(data).to_csv("example/CollisionData.csv")
                        


