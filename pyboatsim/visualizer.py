import trimesh
import numpy as np
from boatsim import BoAtSim
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyboatsim.constants import AXES, EPSILON
import pandas as pd
import tqdm as tqdm

class Visualizer:

    def __init__(self, boatsim:BoAtSim, visualization_model:trimesh.Trimesh):
        self.boatsim = boatsim
        self.visualization_model = visualization_model
        self.data = self.boatsim.get_history_as_dataframe()

    def _update(self, step_idx:int, axes:plt.Axes):
            # Array representation of position & rotation
            state = self.data.iloc[step_idx]
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

            # Prepare the axes and plot
            vertices = visualization_model_temp.vertices
            T = visualization_model_temp.faces
            axes.clear()
            axes.set_xlim(r[0]-2, r[0]+2)
            axes.set_ylim(r[1]-2, r[1]+2)
            axes.set_zlim(r[2]-2, r[2]+2)

            xx, yy = np.meshgrid((r[0]-2, r[0]+2), (r[1]-2, r[1]+2))
            z = state["r_z__water"]*np.ones(xx.shape)
            water = axes.plot_surface(xx, yy, z, color="b", alpha=0.1, zorder=1)
            boat = axes.plot_trisurf(
                vertices[:,0], vertices[:,1], vertices[:,2], 
                triangles = T, edgecolor=[[0,0,0]], linewidth=1.0, 
                alpha=0.5, shade=True, color="k",zorder=1
            )
            return water, boat


    def animate(self, save_path:str=None, verbose:bool=True):

        fig = plt.figure()
        axes = fig.add_subplot(projection='3d')

        dt = self.data["t"].iloc[1] - self.data["t"].iloc[0]
        ani = animation.FuncAnimation(
            fig=fig, 
            func=lambda i, axes: self._update(i, axes), 
            frames=len(self.data), 
            interval=dt*1000, 
            fargs=(axes,)
        )

        with tqdm.tqdm(total=len(self.data)) as bar:
            if save_path is None: plt.show()
            else: ani.save(save_path, progress_callback=lambda i, n: bar.update())
