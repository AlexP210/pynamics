import trimesh
import numpy as np
from boatsim import BoAtSim
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyboatsim.constants import AXES, EPSILON
import pandas as pd

class Visualizer:

    def __init__(self, boatsim:BoAtSim, visualization_model:trimesh.Trimesh):
        self.boatsim = boatsim
        self.visualization_model = visualization_model
        self.data = self.boatsim.get_history_as_dataframe()

    def animate(self):

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        def update(step_idx:int, data:pd.DataFrame):
            # Array representation of position & rotation
            state = data.iloc[step_idx]
            theta = np.array([state[f"theta_{axis}__boat"] for axis in AXES])
            r = np.array([state[f"r_{axis}__boat"] for axis in AXES])
            # Transform the mesh
            translation_matrix = trimesh.transformations.translation_matrix(direction=r)
            if np.linalg.norm(theta) <= EPSILON:
                rotation_matrix = np.eye(4)
            else:
                rotation_matrix = trimesh.transformations.rotation_matrix(
                    direction=theta/np.linalg.norm(theta),
                    angle=np.linalg.norm(theta))
            transformation_matrix = trimesh.transformations.concatenate_matrices(
                    translation_matrix,
                    rotation_matrix
            )

            visualization_model_temp = self.visualization_model.copy()
            visualization_model_temp.apply_transform(transformation_matrix)

            vertices = visualization_model_temp.vertices
            T = visualization_model_temp.faces
            ax.clear()
            ax.set_xlim(r[0]-2, r[0]+2)
            ax.set_ylim(r[1]-2, r[1]+2)
            ax.set_zlim(r[2]-2, r[2]+2)
            xx, yy = np.meshgrid((r[0]-2, r[0]+2), (r[1]-2, r[1]+2))
            z = state["r_z__water"]*np.ones(xx.shape)
            surf = ax.plot_surface(xx, yy, z, color="b", alpha=0.1, zorder=1)
            plotted = ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2], triangles = T, edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.5, shade=True, color="k",zorder=1)
            return plotted, surf

        ts = self.data["t"]
        dt = ts.iloc[1] - ts.iloc[0]
        print(ts.iloc[1] - ts.iloc[0])
        ani = animation.FuncAnimation(fig=fig, func=update, frames=len(self.data), interval=1, fargs=(self.data,))
        plt.show()