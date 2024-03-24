import numpy as np

class Frame:
    def __init__(
        self,
        translation:np.matrix=np.zeros((3,1)),
        rotation:np.matrix=np.eye(3,3)
    ):
        self.translation = translation
        self.rotation = rotation
        self.matrix = np.block([
            [rotation, translation],
            [0,0,0,1]
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
        self.tree = {root_body_name: (root_body_name, "Identity")}
        self.bodies = {root_body_name: root_body}

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

    def get_transform(
            self,
            from_body_name:str,
            from_frame_name:str,
            to_body_name:str,
            to_frame_name:str
    ):
        return np.linalg.inv(self._get_transform_from_root(from_body_name, from_frame_name)) * self._get_transform_from_root(to_body_name, to_frame_name)
    
if __name__ == "__main__":

    wheel_tip_frame = Frame(
        translation=np.matrix([[1,], [0,], [0]]),
        rotation=np.matrix([
            [1, 0, 0],
            [0, -0, -1],
            [0, 1, 0]
        ])
    )
    left_wheel_hub_frame = Frame(
        translation=np.matrix([[0,], [1,], [0]]),
        rotation=np.matrix([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    )
    right_wheel_hub_frame = Frame(
        translation=np.matrix([[0,], [-1,], [0]]),
        rotation=np.matrix([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    )

    boat_body = Body(
        mass=10,
        center_of_mass=np.matrix([[1,], [0,], [0]]),
        inertia_matrix=np.eye(3,3)
    )
    water_wheel_body = Body(
        mass=10,
        center_of_mass=np.matrix([[1,], [0,], [0]]),
        inertia_matrix=np.eye(3,3)
    )

    topology = Topology(
        root_body=boat_body,
        root_body_name="Boat"
    )
    topology.add_frame(body_name="Boat", frame=left_wheel_hub_frame, frame_name="Left Water Wheel Frame")
    topology.add_frame(body_name="Boat", frame=right_wheel_hub_frame, frame_name="Right Water Wheel Frame")
    
    topology.add_connection("Boat", "Left Water Wheel Frame", water_wheel_body, "Left Water Wheel")
    topology.add_connection("Boat", "Right Water Wheel Frame", water_wheel_body, "Right Water Wheel")

    topology.add_frame(body_name="Left Water Wheel", frame=wheel_tip_frame, frame_name="Water Wheel Tip Frame")
    topology.add_frame(body_name="Right Water Wheel", frame=wheel_tip_frame, frame_name="Water Wheel Tip Frame")

    transform1 = topology.get_transform(
        from_body_name="Boat",
        from_frame_name="Identity",
        to_body_name="Left Water Wheel",
        to_frame_name="Water Wheel Tip Frame"
    )
    transform2 = topology.get_transform(
        from_body_name="Left Water Wheel",
        from_frame_name="Water Wheel Tip Frame",
        to_body_name="Boat",
        to_frame_name="Identity"
    )

    transform3 = topology.get_transform(
        from_body_name="Left Water Wheel",
        from_frame_name="Water Wheel Tip Frame",
        to_body_name="Right Water Wheel",
        to_frame_name="Water Wheel Tip Frame"
    )

    print(transform3)



    