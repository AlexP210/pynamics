import numpy as np
from enum import Enum
import typing as types
from pyboatsim.constants import EPSILON
from pyboatsim.kinematics.joint import Joint, FixedJoint, RevoluteJoint
import pyboatsim.math.linalg as linalg

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
    

class Body:

    def __init__(
            self, 
            mass:float=1,
            center_of_mass:np.matrix=np.zeros((3,1)),
            inertia_matrix:np.matrix=np.eye(3,3),
            velocity:np.matrix=np.matrix(np.zeros((6,1))),
            acceleration:np.matrix=np.matrix(np.zeros((6,1))),
        ):
        self.mass = mass
        self.center_of_mass = center_of_mass
        self.inertia_matrix = inertia_matrix
        self.velocity = velocity
        self.acceleration = acceleration
        # Pg. 33
        cx = linalg.R3_cross_product_matrix(self.center_of_mass)
        self.mass_matrix = np.matrix(np.block([
            [self.inertia_matrix+self.mass*cx@cx.T, self.mass*cx],
            [self.mass*cx.T, self.mass*np.eye(3,3)]
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

    def get_velocity(self):
        return self.velocity
    def get_acceleration(self):
        return self.acceleration
    
    def set_velocity(self, velocity):
        self.velocity = velocity
    def set_acceleration(self, acceleration):
        self.acceleration = acceleration
    
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
        self.tree = {}
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
        body_identity_to_frame_transformation = self.bodies[to_body_name].frames[to_frame_name].matrix
        articulation_transformation = self.joints[to_body_name].get_T()

        if to_body_name != "World":
            parent_body_name, parent_frame_name = self.tree[to_body_name]
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
        return np.linalg.inv(self._get_transform_from_root(from_body_name, from_frame_name)) @ self._get_transform_from_root(to_body_name, to_frame_name)
    
    def get_X(
        self,
        from_body_name:str,
        from_frame_name:str,
        to_body_name:str,
        to_frame_name:str
    ):
        T = self.get_transform(from_body_name, from_frame_name, to_body_name, to_frame_name)
        E = T[:3,:3].T
        r = T[:3,3]
        r_cross = linalg.R3_cross_product_matrix(r)
        return np.block([
            [E, np.zeros((3,3))],
            [-E@r_cross, E]
        ])
    def get_Xstar(
        self,
        from_body_name:str,
        from_frame_name:str,
        to_body_name:str,
        to_frame_name:str
    ):
        T = self.get_transform(from_body_name, from_frame_name, to_body_name, to_frame_name)
        E = T[:3,:3].T
        r = T[:3,3]
        r_cross = linalg.R3_cross_product_matrix(r)
        return np.block([
            [E, -E@r_cross],
            [np.zeros((3,3)), E]
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

    def get_mass_matrix(self):
        mass_matrix = {}
        body_names = self.get_ordered_body_list()
        I_Cs = {}
        body_name_to_number = {"World": 0}

        for body_number, body_name in list(enumerate(body_names))[1:]:
            I_Cs[body_name] = self.bodies[body_name].mass_matrix.copy()
            body_name_to_number[body_name] = body_number

        for i, body_name_i in list(enumerate(body_names))[-1:0:-1]:
            parent_body_name_i, _ = self.tree[body_name_i]
            if parent_body_name_i != "World":
                lambda_i_xstar_i = self.get_Xstar(body_name_i, "Identity", parent_body_name_i, "Identity")
                i_x_lambda_i = self.get_X(parent_body_name_i, "Identity", body_name_i, "Identity")
                I_Cs[parent_body_name_i] += lambda_i_xstar_i @ I_Cs[body_name_i] @ i_x_lambda_i
            joint_i = self.joints[body_name_i]
            S_i = joint_i.get_motion_subspace()
            F = I_Cs[body_name_i] @ S_i
            mass_matrix[(body_name_i, body_name_i)] = S_i.T @ F
            body_name_j = body_name_i
            parent_body_name_j, _ = self.tree[body_name_j]
            while parent_body_name_j != "World":
                F = self.get_Xstar(body_name_j, "Identity", parent_body_name_j, "Identity") @ F
                body_name_j = parent_body_name_j
                parent_body_name_j, _ = self.tree[body_name_j]
                n_dof_j = self.joints[body_name_j].get_number_degrees_of_freedom()
                mass_matrix[(body_name_i, body_name_j)] = F.T @ self.joints[body_name_j].get_motion_subspace()
                mass_matrix[(body_name_j, body_name_i)] = mass_matrix[(body_name_i, body_name_j)].T
        return mass_matrix

    def get_ordered_body_list(self):
        if self.body_list is None:
            number_of_parents = []
            for body_name in self.bodies.keys():
                n_of_p = 0
                current_body_name = body_name
                while current_body_name != "World": 
                    current_body_name, _ = self.tree[current_body_name]
                    n_of_p += 1
                number_of_parents.append((body_name, n_of_p))
            sorted_number_of_parents = sorted(number_of_parents, key=lambda t: t[1])
            self.body_list = list(zip(*sorted_number_of_parents))[0]
        return self.body_list
        
    def calculate_body_velocities(self):
        body_velocities = {}
        body_names = self.get_ordered_body_list()
        body_velocities[body_names[0]] = np.matrix(np.zeros(6)).T
        for body_name in body_names[1:]:
            joint = self.joints[body_name]
            parent_body_name, parent_frame_name = self.tree[body_name]
            X_J = self.get_X(parent_body_name, parent_frame_name, body_name, "Identity")
            X_T = self.get_X(parent_body_name, "Identity", parent_body_name, parent_frame_name)
            v_J = joint.get_velocity()
            i__X__lambda_i = X_J @ X_T
            i_X_0 = self.get_X("World", "Identity", body_name, "Identity")
            body_velocities[body_name] = i__X__lambda_i @ self.bodies[parent_body_name].get_velocity() + v_J
        return body_velocities
    
    def update_body_velocities(self):
        for body_name, velocity in self.calculate_body_velocities().items():
            self.bodies[body_name].set_velocity(velocity)

    def calculate_body_accelerations(self, q_dd):
        body_accelerations = {}
        if type(q_dd) == type(np.matrix(0)):
            joint_accelerations = self.dictionarify(q_dd)
        elif type(q_dd) == type(dict()):
            joint_accelerations = q_dd
        body_names = self.get_ordered_body_list()
        # Initialize the states to track velocity & acceleration
        body_accelerations[body_names[0]] = np.matrix(np.zeros(shape=(6,1)))
        for body_name in body_names[1:]:
            joint = self.joints[body_name]
            if body_name in joint_accelerations:
                joint_acceleration = joint_accelerations[body_name]
            else:
                joint_acceleration = np.matrix(np.zeros(shape=(joint.get_number_degrees_of_freedom(),1)))
            parent_body_name, parent_frame_name = self.tree[body_name]
            X_J = self.get_X(parent_body_name, parent_frame_name, body_name, "Identity")
            X_T = self.get_X(parent_body_name, "Identity", parent_body_name, parent_frame_name)
            S_i = joint.get_motion_subspace()
            v_J = joint.get_velocity()
            i__X__lambda_i = X_J @ X_T
            i_X_0 = self.get_X("World", "Identity", body_name, "Identity")
            c_J = joint.get_c()

            a1 = i__X__lambda_i @ body_accelerations[parent_body_name]
            a2 = joint.get_acceleration(joint_acceleration)
            a3 = c_J + linalg.cross(self.bodies[body_name].get_velocity()) @ v_J
            body_accelerations[body_name] = a1 + a2 + a3

        return body_accelerations


    def update_body_accelerations(self):
        for body_name, acceleration in self.calculate_body_accelerations().items():
            self.bodies[body_name].set_acceleration(acceleration)


    def vectorify(self, dictionary):
        body_names = self.get_ordered_body_list()
        dof = [self.joints[body_name].get_number_degrees_of_freedom() for body_name in body_names]
        N = sum(dof)
        vector = np.matrix(np.zeros(shape=(N,1)))
        s = 0
        for i in range(len(body_names)):
            body_name = body_names[i]
            if body_name not in dictionary:
                vector_elements = np.matrix(np.zeros(shape=(dof[i],1)))
            else:
                vector_elements = dictionary[body_names[i]]
            vector[s:s+dof[i],0] = vector_elements
            s+=dof[i]
        return vector
    
    def matrixify(self, dictionary):
        body_names = self.get_ordered_body_list()
        dof = [self.joints[body_name].get_number_degrees_of_freedom() for body_name in body_names]
        N = sum(dof)
        matrix = np.matrix(np.zeros(shape=(N,N)))
        s_i = 0
        for i, body_name_i in enumerate(body_names):
            s_j = 0
            for j, body_name_j in enumerate(body_names):
                if (body_name_i, body_name_j) in dictionary:
                    matrix_elements = dictionary[(body_name_i, body_name_j)]
                else:
                    matrix_elements = np.matrix(np.zeros(shape=(dof[i], dof[j])))
                matrix[s_i:s_i+dof[i], s_j:s_j+dof[j]] = matrix_elements
                s_j += dof[j]
            s_i += dof[i]
        return matrix

    
    def dictionarify(self, vector):
        body_names = self.get_ordered_body_list()
        dof = {body_name: self.joints[body_name].get_number_degrees_of_freedom() for body_name in body_names}
        s = 0
        dictionary = {}
        for body_name in body_names:
            dictionary[body_name] = vector[s:s+dof[body_name],0]
            s += dof[body_name]
        return dictionary

    def get_joint_space_positions(self):
        return {
            body_name: self.joints[body_name].get_configuration()
            for body_name in self.get_ordered_body_list()
        }

    def get_joint_space_velocities(self):
        return {
            body_name: self.joints[body_name].get_configuration_d()
            for body_name in self.get_ordered_body_list()
        }