import pandas as pd
import matplotlib.pyplot as plt

from pyboatsim.state import State
from pyboatsim.dynamics import DynamicsParent, WaterWheel, BodyDrag, ConstantForce
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
            "gamma__waterwheel": 0,
            "gammadot__waterwheel": 0.01,
        }), 
        dynamics=[
            WaterWheel("waterwheel", 1, 1, 0.1, 2, 1, 1),
            BodyDrag("bodydrag", 1, 1),
        ]
    )

    # Run the sim
    sim.simulate(delta_t=3, dt=0.001, verbose=True)
    data = pd.DataFrame.from_dict(sim.history)

    # Plot the results
    plt.plot(data["t"], data["f_x__waterwheel"], label="f_x__waterwheel")
    plt.plot(data["t"], data["f_x__bodydrag"], label="f_x__bodydrag")
    plt.plot(data["t"], data["f_x__total"], label="f_x__total")
    plt.title("Forces During Basic Bo-At Sim")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.legend()
    plt.show()