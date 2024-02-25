import pandas as pd
import matplotlib.pyplot as plt

from pyboatsim.state import State
from pyboatsim.dynamics import DynamicsParent, WaterWheel, SimpleBodyDrag, ConstantForce
from pyboatsim.boatsim import BoAtSim

if __name__ == "__main__":
    # Assemble the sim
    sim = BoAtSim(
        state=State(
            state_dictionary={
            "t": 0,
            "r_x__boat": 0, 
            "r_y__boat": 0,
            "r_z__boat": 0,
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
            "m__boat": 1,
            "I_xx__boat": 1,
            "I_yy__boat": 1,
            "I_zz__boat": 1,
            "rho": 1000,
            "v_x__water": 0,
            "v_y__water": 0, 
            "v_z__water": 0,
            "gamma__waterwheelleft": 0,
            "gammadot__waterwheelleft": 3.14,
            "gamma__waterwheelright": 0,
            "gammadot__waterwheelright": 3.14,
        }), 
        dynamics=[
            WaterWheel(
                name="waterwheelleft",
                wheel_radius=0.1, 
                wheel_hub_height=0.05,
                paddle_width=0.05,
                number_of_paddles_per_wheel=6,
                paddle_drag_coefficient=1.28,
                torque_polarity=+1),
            WaterWheel(
                name="waterwheelright",
                wheel_radius=0.1, 
                wheel_hub_height=0.05,
                paddle_width=0.05,
                number_of_paddles_per_wheel=6,
                paddle_drag_coefficient=1.28,
                torque_polarity=-1),
            SimpleBodyDrag(
                name="bodydrag", 
                cross_sectional_area=0.1*0.01*0.5**(0.5), 
                drag_coefficient=1.28),
        ]
    )

    # Run the sim
    sim.simulate(delta_t=20, dt=0.001, verbose=True)
    data = pd.DataFrame.from_dict(sim.history)

    # Plot the results
    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(data["t"], data["f_x__waterwheelleft"]+data["f_x__waterwheelright"], label="f_x__waterwheels")
    ax[0].plot(data["t"], data["f_x__bodydrag"], label="f_x__bodydrag")
    ax[0].plot(data["t"], data["f_x__total"], label="f_x__total")
    ax[0].legend()
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Force (N)")
    ax[1].plot(data["t"], data["v_x__boat"],label="v__boat")
    ax[1].legend()
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Velocity (m/s)")
    ax[2].plot(data["t"], data["r_x__boat"], label="r__x_boat")
    ax[2].legend()
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Position (m)")
    plt.show()
    plt.show()