import numpy as np
from enum import Enum
import typing as types
from pyboatsim.constants import EPSILON
from pyboatsim.kinematics.joint import Joint, FixedJoint, RevoluteJoint
import pyboatsim.math.linalg as linalg

class Articulation:
    TRANSLATE_X = np.array([1,0,0,0,0,0])
    TRANSLATE_Y = np.array([0,1,0,0,0,0])
    TRANSLATE_Z = np.array([0,0,1,0,0,0])
    ROTATE_X = np.array([0,0,0,1,0,0])
    ROTATE_Y = np.array([0,0,0,0,1,0])
    ROTATE_Z = np.array([0,0,0,0,0,1])
    FREE = np.array([1,1,1,1,1,1])

class ArticulationError(Exception):
    pass

class Frame:
    def __init__(
        self,
        translation:np.matrix=np.matrix([0.0,0.0,0.0]).T,
        rotation:np.matrix=np.eye(3,3)
    ):
        self.translation = translation
        self.rotation = rotation
        self.matrix = np.block([
            [rotation, translation],
            [0,0,0,1]
        ])

    @classmethod
    def get_rotation_matrix(cls, angle, axis):
        c = np.cos(angle)
        s = np.sin(angle)
        C = 1-c
        x = axis[0,0]
        y = axis[1,0]
        z = axis[2,0]
        return np.matrix([
            [x*x*C+c, x*y*C-z*s, x*z*C+y*s],
            [y*x*C+z*s, y*y*C+c, y*z*C-x*s],
            [z*x*C-y*s, z*y*C+x*s, z*z*C+c]
        ])
    
    def get_X(self):
        E = self.rotation
        r = self.translation
        r_cross = linalg.cross_product_matrix(r)
        return np.block([
            [E, np.zeros((3,3))],
            [-E@r_cross, E]
        ])

class Body:

    def __init__(
            self, 
            mass:float=1,
            center_of_mass:np.matrix=np.zeros((3,1)),
            inertia_matrix:np.matrix=np.eye(3,3)
        ):
        self.mass = mass
        self.center_of_mass = center_of_mass
        self.inertia_matrix = inertia_matrix
        # Pg. 33
        self.mass_matrix = np.matrix(np.block([
            [self.inertia_matrix, self.mass*linalg.R3_cross_product_matrix(self.center_of_mass)],
            [self.mass*linalg.R3_cross_product_matrix(self.center_of_mass).T, self.mass*np.eye(3,3)]
        ]))
        self.frames = {
            "Identity": Frame(),
            "Center of Mass": Frame(translation=center_of_mass)
        }
    def _assert_frame_in_body(self, frame_name:str):
        if not frame_name in self.frames:
            raise ValueError(
                f"A frame with the name {frame_name} is not found in the body."
            )
    def _assert_frame_not_in_body(self, frame_name:str):
        if frame_name in self.frames:
            raise ValueError(
                f"A frame with the name {frame_name} is already found in the body"
            )
    def add_frame(
            self,
            frame:Frame,
            frame_name:str
        ):
        self.frames[frame_name] = frame

    def get_X(self, frame_name):
        self.frames[frame_name].get_X()

    def copy(self):
        body_copy =  Body(
            mass=self.mass, 
            center_of_mass=self.center_of_mass,
            inertia_matrix=self.inertia_matrix
            )
        body_copy.frames = self.frames
        return body_copy

class Topology:

    def __init__(
            self, 
    ):
        self.tree = {"World": ("World", "Identity")}
        self.bodies = {
            "World": Body(
                mass=0,
                inertia_matrix=np.matrix(np.zeros((3,3))))
            }
        self.joints = {
            "World": FixedJoint()
        }
        self.body_list = None
        self.mass = None
        self.center_of_mass = None
        self.inertia_tensor = None

        # Initialize the mass properties
        self.get_mass()
        self.get_center_of_mass()
        self.get_inertia_tensor()

    def add_frame(
            self,
            body_name:str,
            frame:Frame,
            frame_name:str
    ):
        self._assert_body_in_topology(body_name=body_name)
        self.bodies[body_name].add_frame(frame=frame, frame_name=frame_name)

    def _get_transform_from_root(
            self,
            to_body_name:str,
            to_frame_name:str
    ):  
        # Get the transformation for the articulation
        articulation_transformation = self.joints[to_body_name].get_T()

        body_identity_to_frame_transformation = self.bodies[to_body_name].frames[to_frame_name].matrix
        parent_body_name, parent_frame_name = self.tree[to_body_name]
        if parent_body_name != to_body_name: 
            return self._get_transform_from_root(parent_body_name, parent_frame_name) * articulation_transformation * body_identity_to_frame_transformation
        else:
            return articulation_transformation * body_identity_to_frame_transformation
        
    def _assert_body_in_topology(self, body_name:str):
        if not body_name in self.bodies:
            raise ValueError(
                f"A body with the name {body_name} is not found in "
                "the topology."
            )
    def _assert_body_not_in_topology(self, body_name:str):
        if body_name in self.bodies:
            raise ValueError(
                f"A body with the name {body_name} is already found in"
                "the topology."
            )


    def add_connection(
            self, 
            parent_body_name:str, 
            parent_frame_name:str,
            child_body:Body,
            child_body_name:str,
            joint:Joint,
    ):
        self._assert_body_in_topology(parent_body_name)
        self._assert_body_not_in_topology(child_body_name)
        self.bodies[parent_body_name]._assert_frame_in_body(parent_frame_name)

        self.bodies[child_body_name] = child_body
        self.tree[child_body_name] = (parent_body_name, parent_frame_name)
        self.joints[child_body_name] = joint

        # Mass properties is no longer valid
        self.mass = None
        self.center_of_mass = None
        self.inertia_tensor = None

        # Body list is no longer valid
        self.body_list = None

    def get_transform(
            self,
            from_body_name:str,
            from_frame_name:str,
            to_body_name:str,
            to_frame_name:str
    ):
        return np.linalg.inv(self._get_transform_from_root(from_body_name, from_frame_name)) * self._get_transform_from_root(to_body_name, to_frame_name)
    
    def get_X(
        self,
        from_body_name:str,
        from_frame_name:str,
        to_body_name:str,
        to_frame_name:str
    ):
        T = self.get_transform(from_body_name, from_frame_name, to_body_name, to_frame_name)
        E = T[:3,:3]
        r = T[:3,3]
        r_cross = linalg.cross_product_matrix(r)
        return np.block([
            [E, np.zeros((3,3))],
            [-E@r_cross, E]
        ])

    def get_mass(self):
        if self.mass is not None: return self.mass
        return sum([body.mass for body in self.bodies.values()])

    def get_center_of_mass(self, as_matrix:bool=False):
        # If we've already calculated it, return it
        if self.center_of_mass is not None: 
            com_m = self.center_of_mass 
        # Otherwise calculate it
        else:        
            first_mass_moment = np.matrix([0.0, 0.0, 0.0]).T
            total_mass = 0
            for body_name, body in self.bodies.items():
                base_to_com_matrix = self.get_transform(
                    from_body_name="World",
                    from_frame_name="Identity",
                    to_body_name=body_name,
                    to_frame_name="Center of Mass"
                )
                base_to_com_translation = base_to_com_matrix[0:3, 3]
                total_mass += body.mass
                first_mass_moment += body.mass * base_to_com_translation
            self.mass = total_mass
            if self.mass > 0:
                self.center_of_mass = first_mass_moment / self.mass
            else:
                self.center_of_mass = np.matrix([0.0, 0.0, 0.0]).T

            com_m = self.center_of_mass

        if not as_matrix: 
            return com_m
        else:
            ret = np.matrix(np.eye(4,4))
            ret[0:3,3] = com_m[:,0]
            return ret

    def get_inertia_tensor(self):
        if self.inertia_tensor is not None: return self.inertia_tensor

        base_frame_to_topo_com_frame = self.get_center_of_mass(as_matrix=True)
        topology_inertia_tensor = np.zeros(shape=(3,3))
        for body_name, body in self.bodies.items():
            body_com_frame_to_base_frame = self.get_transform(
                from_body_name=body_name,
                from_frame_name="Center of Mass",
                to_body_name="World",
                to_frame_name="Identity"
            )
            A = body_com_frame_to_base_frame * base_frame_to_topo_com_frame
            R = A[0:3, 0:3]
            T = A[0:3,3]
            rotated_body_inertia_tensor = R @ body.inertia_matrix @ R.T
            correction = body.mass*( (np.linalg.norm(T)**2)*np.matrix(np.eye(3,3)) - T@T.T )
            transformed_body_inertia_tensor = rotated_body_inertia_tensor + correction

            topology_inertia_tensor += transformed_body_inertia_tensor

        self.inertia_tensor = topology_inertia_tensor
        return self.inertia_tensor

    def get_ordered_body_list(self):
        if self.body_list is None:
            number_of_parents = []
            for body_name in self.bodies.keys():
                n_of_p = 0
                current_body_name = body_name
                while current_body_name != "World": 
                    current_body_name, _ = self.bodies[current_body_name]
                    n_of_p += 1
                number_of_parents.append((body_name, n_of_p))
            sorted_number_of_parents = sorted(number_of_parents, key=lambda t: t[1])
            self.body_list = zip(*sorted_number_of_parents)[0]
        return self.body_list
        

if __name__ == "__main__":

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
    base = body.copy()
    roll_body = body.copy()
    pitch_body_1 = body.copy()
    pitch_body_2 = body.copy()
    yaw_body = body.copy()
    
    base.add_frame(base_mounting_point, "Base to Roll Body")
    roll_body.add_frame(short_end, "Roll Body to Pitch Body 1")
    pitch_body_1.add_frame(long_end, "Pitch Body 1 to Pitch Body 2")
    pitch_body_2.add_frame(long_end_for_yaw, "Pitch Body 2 to Yaw Body")
    yaw_body.add_frame(short_end, "End Effector")


    robot = Topology()
    robot.add_connection("World", "Identity", base, "Base Body")
    robot.add_connection(
        "Base Body", "Base to Roll Body", roll_body, "Roll Body",
        joint=RevoluteJoint(0))
    robot.add_connection(
        "Roll Body", "Roll Body to Pitch Body 1", pitch_body_1, "Pitch Body 1",
        joint=RevoluteJoint(0))
    robot.add_connection(
        "Pitch Body 1", "Roll Body to Pitch Body 1", pitch_body_2, "Pitch Body 2",
        joint=RevoluteJoint(0))
    robot.add_connection(
        "Pitch Body 2", "Pitch Body 2 to Yaw Body", yaw_body, "Yaw Body",
        joint=RevoluteJoint(0))
    