import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from pyboatsim.state import State
from pyboatsim.dynamics import DynamicsParent, WaterWheel, SimpleBodyDrag, ConstantForce, MeshBuoyancy, MeshGravity, MeshBodyDrag
from pyboatsim.topology import Topology, Frame, Body
from pyboatsim.boatsim import BoAtSim
from pyboatsim.constants import AXES
from pyboatsim.visualizer import Visualizer

if __name__ == "__main__":
    # Define the bodies
    boat_body = Body(
        mass=1000,
        center_of_mass=np.zeros(shape=(3,1)),
        inertia_matrix=np.matrix([
            [(1000/12)*(2**2 + 0.4**2), 0, 0],
            [0, (1000/12)*(2**2 + 0.4**2), 0],
            [0, 0, (1000/12)*(2**2 + 2**2)]
        ])
    )
    inertia_matrix_com=np.matrix([
        [(1/12)*(0.01**2 + 0.5**2), 0, 0],
        [0, (1/12)*(1**2 + 0.01**2), 0],
        [0, 0, (1/12)*(1**2 + 0.5**2)]
    ])
    T = np.matrix([0.5, 0.0, 0.0]).T
    correction = 1*( (np.linalg.norm(T)**2)*np.matrix(np.eye(3,3)) - T@T.T )
    inertia_matrix_com += correction
    paddle_body = Body(
        mass=1,
        center_of_mass=np.matrix([0.5,0.0,0.0]).T,
        inertia_matrix=inertia_matrix_com
    )

    # Create the topology
    left_wheel_hub_frame = Frame(translation=np.matrix([0, 1.25, 0]).T)
    right_wheel_hub_frame = Frame(translation=np.matrix([0, -1.25, 0]).T)
    boat_body.add_frame(left_wheel_hub_frame, "Left Wheel Hub Frame")
    boat_body.add_frame(right_wheel_hub_frame, "Right Wheel Hub Frame")
    boat = Topology(root_body=boat_body, root_body_name="Boat Body")
    
    left_wheel_frame = Frame(
        translation=np.matrix([0, 1.25, 0]).T,
    )
    right_wheel_frame = Frame(
        translation=np.matrix([0, -1.25, 0]).T,
    )

    boat.add_frame("Boat Body", left_wheel_frame, f"Left Wheel Frame")
    boat.add_frame("Boat Body", right_wheel_frame, f"Right Wheel Frame")
    boat.add_connection(
        parent_body_name="Boat Body", 
        parent_frame_name=f"Left Wheel Frame", 
        child_body=paddle_body, 
        child_body_name=f"Left Wheel")
    boat.add_connection(
        parent_body_name="Boat Body", 
        parent_frame_name=f"Right Wheel Frame", 
        child_body=paddle_body, 
        child_body_name=f"Right Wheel")
        
    # Assemble the sim
    sim = BoAtSim(
        state=State(
            state_dictionary={
            "t": 0,
            "r_x__boat": 0, 
            "r_y__boat": 0,
            "r_z__boat": 2,
            "r_z__water": 0,
            "v_x__boat": 0,
            "v_y__boat": 0, 
            "v_z__boat": 0,
            "a_x__boat": 0, 
            "a_y__boat": 0, 
            "a_z__boat": 0, 
            "theta_x__boat": 0, 
            "theta_y__boat": 0, 
            "theta_z__boat": 0, 
            "omega_x__boat": 0, 
            "omega_y__boat": 0, 
            "omega_z__boat": 0,
            "alpha_x__boat": 0, 
            "alpha_y__boat": 0, 
            "alpha_z__boat": 0,
            "m__boat": boat.get_mass(),
            "I_xx__boat": boat.get_inertia_tensor()[0,0],
            "I_xy__boat": boat.get_inertia_tensor()[0,1],
            "I_xz__boat": boat.get_inertia_tensor()[0,2],
            "I_yx__boat": boat.get_inertia_tensor()[1,0],
            "I_yy__boat": boat.get_inertia_tensor()[1,1],
            "I_yz__boat": boat.get_inertia_tensor()[1,2],
            "I_zx__boat": boat.get_inertia_tensor()[2,0],
            "I_zy__boat": boat.get_inertia_tensor()[2,1],
            "I_zz__boat": boat.get_inertia_tensor()[2,2],
            "c_x__boat": boat.get_center_of_mass()[0,0],
            "c_y__boat": boat.get_center_of_mass()[1,0],
            "c_z__boat": boat.get_center_of_mass()[2,0],
            "rho__water": 1000,
            "v_x__water": 0,
            "v_y__water": 0, 
            "v_z__water": 0,
            "gamma__waterwheel": 0,
            "gammadot__waterwheel": 0.1,
        }), 
        dynamics=[
            MeshBuoyancy(
                name="buoyancy", 
                buoyancy_model_path="/home/alex/Projects/PyBoAtSim/models/cup/cup_boundary.obj"
            ),
            MeshGravity(
                name="gravity", 
                gravity_model_path="/home/alex/Projects/PyBoAtSim/models/cup/cup.obj"
            ),
            MeshBodyDrag(
                name="bodydrag",
                bodydrag_model_path="/home/alex/Projects/PyBoAtSim/models/cup/cup_boundary_poked.obj"
            )
        ],
        topology=boat
    )

    # Run the sim
    print("Running simulation")
    sim.simulate(delta_t=10, dt=0.01, verbose=True)
    data = pd.DataFrame.from_dict(sim.history)

    fig, ax = plt.subplots(nrows=2, ncols=3)
    for row_idx, position_orientation in enumerate(["r", "theta"]):
        for col_idx, axis in enumerate(AXES):
            ax[row_idx, col_idx].plot(
                data["t"], 
                data[f"{position_orientation}_{axis}__boat"], 
                label=f"{position_orientation}_{axis}__boat"
            )
            ax[row_idx, col_idx].set_xlabel("Time (s)")
            if position_orientation == "r": ylabel = "Position (m)"
            elif position_orientation == "theta": ylabel = "Angle (rad)"
            ax[row_idx, col_idx].set_ylabel(ylabel)
            ax[row_idx, col_idx].legend()
    plt.show()

    vis_models = {
        ("Boat Body", "Identity"): trimesh.load(
            file_obj="/home/alex/Projects/PyBoAtSim/models/cup/cup.obj", 
            file_type="obj", 
            force="mesh"
        ),
        ("Left Wheel", "Identity"): trimesh.load(
            file_obj="/home/alex/Projects/PyBoAtSim/models/cup/water_wheel.obj", 
            file_type="obj", 
            force="mesh"
        ),
        ("Right Wheel", "Identity"): trimesh.load(
            file_obj="/home/alex/Projects/PyBoAtSim/models/cup/water_wheel.obj", 
            file_type="obj", 
            force="mesh"
        ),
    }

    vis = Visualizer(boatsim=sim, visualization_models=vis_models)
    print("Saving animation")
    vis.animate(save_path="Test.mp4", show_forces=True)
