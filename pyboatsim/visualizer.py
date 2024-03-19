import trimesh
import numpy as np
from boatsim import BoAtSim
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyboatsim.constants import AXES, EPSILON
import pandas as pd
import tqdm as tqdm
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
        
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)

class Visualizer:

    def __init__(self, boatsim:BoAtSim, visualization_model:trimesh.Trimesh):
        self.boatsim = boatsim
        self.visualization_model = visualization_model
        self.data = self.boatsim.get_history_as_dataframe()
        self.force_magnitudes = {}
        self.torque_magnitudes = {}
        for dynamics_source in self.boatsim.dynamics:
            self.force_magnitudes[dynamics_source.name] = np.linalg.norm(
                x=self.data[[f"f_{axis}__{dynamics_source.name}" for axis in AXES]],
                axis=1
            ).max()
            self.torque_magnitudes[dynamics_source.name] = np.linalg.norm(
                x=self.data[[f"tau_{axis}__{dynamics_source.name}" for axis in AXES]],
                axis=1
            ).max()

    def _update(self, step_idx:int, axes:plt.Axes, show_forces:bool):
        # Array representation of position & rotation
        state = self.data.iloc[step_idx]
        theta = np.array([state[f"theta_{axis}__boat"] for axis in AXES])
        r = np.array([state[f"r_{axis}__boat"] for axis in AXES])
        c = np.array([state[f"c_{axis}__boat"] for axis in AXES])
        com__worldframe = r+c
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
        if show_forces:
            for dynamics_source in self.boatsim.dynamics:
                force__worldframe_m = np.matrix([
                    [state[f"f_{axis}__{dynamics_source.name}"],]
                    for axis in AXES
                ])
                if self.force_magnitudes[dynamics_source.name] > EPSILON:
                    force__worldframe_m/=self.force_magnitudes[dynamics_source.name]
                force_arrow = Arrow3D(
                    xs=[com__worldframe[0],com__worldframe[0]+force__worldframe_m[0,0]],
                    ys=[com__worldframe[1],com__worldframe[1]+force__worldframe_m[1,0]],
                    zs=[com__worldframe[2],com__worldframe[2]+force__worldframe_m[2,0]], 
                    arrowstyle="->",
                    linewidth=5,
                    color="r",
                )
                axes.add_artist(force_arrow)
                tau__comframe_m = np.matrix([
                    [state[f"tau_{axis}__{dynamics_source.name}"],]
                    for axis in AXES
                ])
                tau__worldframe_m = rotation_matrix[:3,:3]@tau__comframe_m
                if self.torque_magnitudes[dynamics_source.name] > EPSILON:
                    tau__worldframe_m/=self.torque_magnitudes[dynamics_source.name]
                torque_arrow = Arrow3D(
                    xs=[com__worldframe[0],com__worldframe[0]+tau__worldframe_m[0,0]],
                    ys=[com__worldframe[1],com__worldframe[1]+tau__worldframe_m[1,0]],
                    zs=[com__worldframe[2],com__worldframe[2]+tau__worldframe_m[2,0]], 
                    arrowstyle="->",
                    linewidth=5,
                    color="b",
                )
                axes.add_artist(torque_arrow)
        return water, boat


    def animate(self, save_path:str=None, verbose:bool=True, figsize=(12,10), show_forces:bool=False):

        fig = plt.figure(figsize=figsize)
        axes = fig.add_subplot(projection='3d')

        dt = self.data["t"].iloc[1] - self.data["t"].iloc[0]
        ani = animation.FuncAnimation(
            fig=fig, 
            func=lambda i, axes, show_forces: self._update(i, axes, show_forces), 
            frames=len(self.data), 
            interval=dt*1000, 
            fargs=(axes,show_forces)
        )

        if save_path is None: 
            plt.show()
        else:
            with tqdm.tqdm(total=len(self.data)) as bar:
                ani.save(save_path, progress_callback=lambda i, n: bar.update())
                plt.clf()
                plt.cla()
                plt.close()