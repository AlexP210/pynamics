import typing

import numpy as np
import scipy.integrate as integrate
import trimesh

from pyboatsim.dynamics import DynamicsParent
from pyboatsim.state import State
from pyboatsim.constants import AXES, EPSILON

class SimpleBodyDrag(DynamicsParent):
    def __init__(
            self,
            name: str,
            cross_sectional_area: float,
            drag_coefficient: float,
        ):
        super().__init__(name=name)
        self.dynamics_parameters = {
            "cross_sectional_area": cross_sectional_area,
            "drag_coefficient": drag_coefficient,
        }
        self.name = f"bodydrag"

    def required_state_labels(self):
        return [
            "rho" 
            ] + [
                f"v_{axis}__boat" for axis in AXES
            ] + [
                f"v_{axis}__water" for axis in AXES
            ]
    
    def compute_dynamics(self, state:State, dt:float) -> State:
        for axis in AXES:
            factors = [
                np.sign(state[f"v_{axis}__water"] - state[f"v_{axis}__boat"]),
                0.5*self.dynamics_parameters["drag_coefficient"]*state["rho"],
                self.dynamics_parameters["cross_sectional_area"],
                (state[f"v_{axis}__water"] - state[f"v_{axis}__boat"])**2
            ]
            state.set({f"f_{axis}__{self.name}": np.prod(factors)})
            state.set({f"tau_{axis}__{self.name}": 0})
        return state

class MeshBodyDrag(DynamicsParent):
    def __init__(
            self,
            name: str,
            bodydrag_model_path: str,
            drag_coefficient:float=1.28
        ):
        super().__init__(name=name)
        self.dynamics_parameters = {
            "bodydrag_model_path": bodydrag_model_path,
            "drag_coefficient": drag_coefficient
        }
        self.bodydrag_model:trimesh.Trimesh = trimesh.load(
            file_obj=bodydrag_model_path, 
            file_type=bodydrag_model_path.split(".")[-1], 
            force="mesh"
        )
        self.bodydrag_model.merge_vertices()
        self.bodydrag_model.fix_normals()
        if not self.bodydrag_model.is_watertight:
            raise ValueError("BodyDrag volume mesh is not watertight.")

    def required_state_labels(self):
        return [
                "r_z__water",
            ] + [
                f"v_{axis}__water" for axis in AXES
            ] + [
                f"r_{axis}__boat" for axis in AXES
            ] + [
                f"theta_{axis}__boat" for axis in AXES
            ] + [
                "rho__water"
            ]
    
    def compute_dynamics(self, state:State, dt:float) -> State:

        # Array representation of position & rotation
        theta = np.array([state[f"theta_{axis}__boat"] for axis in AXES])
        r = np.array([state[f"r_{axis}__boat"] for axis in AXES])

        # Transform the mesh
        translation_matrix = trimesh.transformations.translation_matrix(direction=r)
        if np.linalg.norm(theta) <= EPSILON:
            rotation_matrix = np.eye(4)
        else:
            rotation_matrix = trimesh.transformations.rotation_matrix(
                direction=theta/np.linalg.norm(theta),
                angle=np.linalg.norm(theta)
            )
        transformation_matrix = trimesh.transformations.concatenate_matrices(
                translation_matrix,
                rotation_matrix
        )

        # Create a temp body drag model & transform
        bodydrag_model_temp:trimesh.Trimesh = self.bodydrag_model.copy()
        bodydrag_model_temp.apply_transform(
            matrix=transformation_matrix
        )

        # Get the submerged section
        submerged:trimesh.Trimesh = bodydrag_model_temp.slice_plane(
            plane_origin=(0,0,0),
            plane_normal=(0,0,-1),
            cap=True
        )
        submerged.merge_vertices()
        submerged.fix_normals()

        # Matrix representations of position & rotation
        theta_m = np.matrix(theta).T
        r_m = np.matrix(r).T
        c_m = np.matrix([
            [state["c_x__boat"],],
            [state["c_y__boat"],],
            [state["c_z__boat"],],
        ])

        # Calculate drag
        total_bodydrag_force__comframe_m = np.matrix([0.0, 0.0, 0.0]).T
        total_bodydrag_torque__comframe_m = np.matrix([0.0, 0.0, 0.0]).T
        v__boat__worldframe_m = np.matrix([
            [state["v_x__boat"],],
            [state["v_y__boat"],],
            [state["v_z__boat"],],
        ])
        omega__boat__comframe_m = np.matrix([
            [state["omega_x__boat"],],
            [state["omega_y__boat"],],
            [state["omega_z__boat"],],
        ])
        v__water__worldframe_m = np.matrix([
            [state["v_x__water"],],
            [state["v_y__water"],],
            [state["v_z__water"],],
        ])
        for face, normal in zip(submerged.faces, submerged.face_normals):
            # Initialize force added by this face to zero
            force__comframe_m = np.matrix([0.0, 0.0, 0.0]).T
            torque__comframe_m = np.matrix([0.0, 0.0, 0.0]).T

            # Get some useful quantities
            normal_m = np.matrix(normal).T
            vertices = [submerged.vertices[face[i]] for i in range(len(face))]
            triangle_center__worldframe = (1/3)*sum(vertices)

            triangle_center__worldframe_m = np.matrix(triangle_center__worldframe).T
            com__worldframe_m = r_m + c_m

            # Compute the velocity of the triangle center in COM frame
            com_to_triangle_center__worldframe_m = triangle_center__worldframe_m - com__worldframe_m
            com_to_triangle_center__comframe_m = np.dot(rotation_matrix.T[0:3, 0:3], com_to_triangle_center__worldframe_m)
            v__boat__comframe_m = np.dot(rotation_matrix.T[0:3, 0:3], v__boat__worldframe_m)
            v__triangle__comframe_m = v__boat__comframe_m + np.cross(omega__boat__comframe_m.T, com_to_triangle_center__comframe_m.T).T
            
            # Get the velocity of the water in COM Frame
            v__water__comframe_m = np.dot(rotation_matrix.T[0:3, 0:3], v__water__worldframe_m)
            
            # Get the relative velocity in the COM Frame
            v_m = v__water__comframe_m - v__triangle__comframe_m
            if np.linalg.norm(v_m) < EPSILON: v_hat_m = np.matrix([0,0,0]).T
            else: v_hat_m = v_m / np.linalg.norm(v_m)
            
            # Check the normals
            normal__comframe_m = np.dot(rotation_matrix.T[0:3, 0:3], normal_m)
            if np.linalg.norm(normal__comframe_m) < EPSILON: normal__comframe_m = np.matrix([0,0,0]).T
            else: normal__comframe_m = normal__comframe_m / np.linalg.norm(normal__comframe_m)
            
            # Normals are oriented outwards. So if the drag force is not
            # directed into the face, no drag contributed by this face
            if np.dot(v_hat_m.T, normal__comframe_m) > 0: continue

            # Calculate the area perpendicular to the velocity
            A = 0.5*np.linalg.norm(
                np.cross(vertices[1]-vertices[0], vertices[2]-vertices[1])
            )
            A_perp = A*float(np.dot(-normal__comframe_m.T, v_hat_m))

            # Calculate the force magnitude
            v_squared = float(np.dot(v_m.T, v_m))
            force_magnitude = 0.5*state["rho__water"]*self.dynamics_parameters["drag_coefficient"]
            force_magnitude*=v_squared*A_perp
            force__comframe_m = force_magnitude*v_hat_m
            point_of_application__comframe_m = com_to_triangle_center__comframe_m
            torque__comframe_m = np.cross(
                point_of_application__comframe_m.T,
                force__comframe_m.T
            ).T
            total_bodydrag_force__comframe_m += force__comframe_m
            total_bodydrag_torque__comframe_m += torque__comframe_m
        total_bodydrag_force__worldframe_m = np.dot(rotation_matrix[0:3, 0:3], total_bodydrag_force__comframe_m)

        # Update the state dict
        for axis_idx in range(len(AXES)):
            axis = AXES[axis_idx]
            state.set({
                f"f_{axis}__{self.name}": total_bodydrag_force__worldframe_m[axis_idx, 0],
                f"tau_{axis}__{self.name}": total_bodydrag_torque__comframe_m[axis_idx, 0]
            })

        return state