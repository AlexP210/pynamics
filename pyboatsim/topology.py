import numpy as np

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
            inertia_matrix:np.matrix=np.eye(3,3)
        ):
        self.mass = mass
        self.center_of_mass = center_of_mass
        self.inertia_matrix = inertia_matrix
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


class Topology:

    def __init__(
            self, 
            root_body:Body,
            root_body_name:str,
    ):
        self.root_body_name = root_body_name
        self.tree = {root_body_name: (root_body_name, "Identity")}
        self.bodies = {root_body_name: root_body}
        self.mass = None
        self.center_of_mass = None
        self.inertia_tensor = None

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
        last_matrix = self.bodies[to_body_name].frames[to_frame_name].matrix
        parent_body_name, parent_frame_name = self.tree[to_body_name]
        if parent_body_name != to_body_name: 
            return self._get_transform_from_root(parent_body_name, parent_frame_name) * last_matrix
        else:
            return last_matrix
        
    def _assert_body_in_topology(self, body_name:str):
        if not body_name in self.bodies:
            raise ValueError(
                f"A body with the name {body_name} is not found in"
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
            child_body_name:str
    ):
        self._assert_body_in_topology(parent_body_name)
        self._assert_body_not_in_topology(child_body_name)
        self.bodies[parent_body_name]._assert_frame_in_body(parent_frame_name)

        self.bodies[child_body_name] = child_body
        self.tree[child_body_name] = (parent_body_name, parent_frame_name)

        # Mass properties is no longer valid
        self.mass = None
        self.center_of_mass = None
        self.inertia_tensor = None

    def get_transform(
            self,
            from_body_name:str,
            from_frame_name:str,
            to_body_name:str,
            to_frame_name:str
    ):
        return np.linalg.inv(self._get_transform_from_root(from_body_name, from_frame_name)) * self._get_transform_from_root(to_body_name, to_frame_name)
    
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
                    from_body_name=self.root_body_name,
                    from_frame_name="Identity",
                    to_body_name=body_name,
                    to_frame_name="Center of Mass"
                )
                base_to_com_translation = base_to_com_matrix[0:3, 3]
                total_mass += body.mass
                first_mass_moment += body.mass * base_to_com_translation
            self.mass = total_mass
            self.center_of_mass = first_mass_moment / self.mass

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
                to_body_name=self.root_body_name,
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


if __name__ == "__main__":

    side = Body(
        mass=1,
        center_of_mass=np.matrix([0.5,0.0,0.0]).T,
        inertia_matrix=np.matrix([
            [0, 0, 0],
            [0, 1/12, 0],
            [0, 0, 1/12]
        ])
    )
    corner = Frame(
        translation=np.matrix([1.0,0.0,0.0]).T, 
        rotation=Frame.get_rotation_matrix(np.pi/2, np.matrix([0.0,0.0,1.0]).T)
    )
    side.add_frame(frame=corner, frame_name="Corner")

    square = Topology(root_body=side, root_body_name="Side 1")
    square.add_connection("Side 1", "Corner", side, "Side 2")
    square.add_connection("Side 2", "Corner", side, "Side 3")
    square.add_connection("Side 3", "Corner", side, "Side 4")

    transform = square.get_transform(
        from_body_name="Side 1",
        from_frame_name="Identity",
        to_body_name="Side 4",
        to_frame_name="Corner"
    )
    print(transform)
    print(square.get_mass())
    print(square.get_center_of_mass())
    print(np.matrix([0.0,0.0,1.0])@square.get_inertia_tensor()@np.matrix([0.0,0.0,1.0]).T)