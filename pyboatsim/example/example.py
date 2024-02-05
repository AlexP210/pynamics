import numpy as np
import matplotlib.pyplot as plt

import pyboatsim.boatsim

# See the effect of R and h/R on the terminal velocity
sim = pyboatsim.BoAtSim()
resolution = 100
h_over_Rs = np.linspace(start=0.1, stop=0.9, num=resolution)
Rs = np.linspace(start=0.01, stop=0.1, num=resolution)
terminal_velocities = np.zeros(shape=(Rs.size, h_over_Rs.size))

for R_idx in range(terminal_velocities.shape[0]):
    for h_over_R_idx in range(terminal_velocities.shape[1]):
        R = Rs[R_idx]
        h_over_R = h_over_Rs[h_over_R_idx]
        sim = pyboatsim.BoAtSim()
        sim.update_state({"h":h_over_R * R, "R": R})
        sim.simulate(5, verbose=True)
        terminal_velocities[R_idx, h_over_R_idx] = sim.history["v_boat"].max()
plt.imshow(terminal_velocities, interpolation="none", cmap="plasma")
label_density = int(resolution/10)
plt.xlabel("Height of Water Wheel / Radius of Water Wheel")
plt.xticks(ticks=range(0, len(h_over_Rs), 1), labels=h_over_Rs[::1].round(2))
plt.ylabel("Radius of Water Wheel (m)")
plt.yticks(ticks=range(0, len(Rs), 1), labels=Rs[::1].round(2))
cbar = plt.colorbar()
cbar.set_label("Velocity After 5s (m/s)")
plt.title("Optimal Paddles", fontsize=20)
plt.savefig("Experiments/Experiment0.png", dpi=300)
