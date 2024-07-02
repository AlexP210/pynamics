import trimesh
import numpy as np
from pyboatsim.boatsim import Sim
from pyboatsim.kinematics.topology import Topology, Body, Frame
from pyboatsim.kinematics.joint import RevoluteJoint, FixedJoint
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyboatsim.constants import AXES, EPSILON
import pandas as pd
import tqdm as tqdm
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import typing

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

    def __init__(
            self, 
            topology:Topology,
            visualization_models:typing.Dict[typing.Tuple[str,str], trimesh.Trimesh],
            sim:Sim=None, 
        ):
        self.topology=topology
        self.visualization_models = visualization_models
        if sim is not None: self.add_sim_data(sim=sim)

    def add_sim_data(self, sim:Sim):
        self.sim = sim

    def _update(self, step_idx:int, axes:plt.Axes):
        # Get the state for this time
        q = self.sim.joint_space_position_history[step_idx]
        q_d = self.sim.joint_space_velocity_history[step_idx]

        axes.clear()

        axes, artists = self._plot(axes, joint_configuration=q)

        return artists

    def view(self, save_path:str=None, verbose:bool=True, figsize=(12,10)):

        fig = plt.figure(figsize=figsize)
        axes = fig.add_subplot(projection='3d')

        axes, artists = self._plot(axes)

        if save_path is None: 
            plt.show()
        else:
            plt.savefig(save_path, dpi=300)
            plt.clf()
            plt.cla()
            plt.close()

    def _plot(self, axes:plt.Axes, joint_configuration=None):

        # Configure the topology based on the provided joint configuration
        if joint_configuration != None:
            for body_name, q in joint_configuration.items():
                self.topology.joints[body_name].set_configuration(q)

        # Plot the topology, and get a bounding box for the plot
        lower_bound = np.zeros(shape=(3,))
        upper_bound = np.zeros(shape=(3,))
        artists = []
        for (body_name, frame_name), visualization_model in self.visualization_models.items():
            visualization_model_temp = visualization_model.copy()
            visualization_model_temp.apply_transform(
                np.array(self.topology.get_transform(
                    from_body_name="World",
                    from_frame_name="Identity",
                    to_body_name=body_name,
                    to_frame_name=frame_name
                ))
            )
            # Prepare the axes and plot
            vertices = visualization_model_temp.vertices
            lb = vertices.min(axis=0)
            ub = vertices.max(axis=0)
            for i in range(3):
                lower_bound[i] = min(lower_bound[i], lb[i])
                upper_bound[i] = max(upper_bound[i], ub[i])

            T = visualization_model_temp.faces

            artist = axes.plot_trisurf(
                vertices[:,0], vertices[:,1], vertices[:,2], 
                triangles = T, edgecolor=[[0,0,0]], linewidth=1.0, 
                alpha=0.5, shade=True, color="k",zorder=1
            )
            artists.append(artist)

        center = 0.5 * (upper_bound + lower_bound)
        max_extent = 0
        for i in range(3):
            max_extent = max(max_extent, 0.5*(upper_bound[i] - lower_bound[i]))
        max_extent *= 1.1
        axes.set_xlim(center[0] - max_extent, center[0] + max_extent)
        axes.set_ylim(center[1] - max_extent, center[1] + max_extent)
        axes.set_zlim(center[2] - max_extent, center[2] + max_extent)

        return axes, artists


    def animate(self, dt, save_path:str=None, verbose:bool=True, figsize=(12,10)):

        fig = plt.figure(figsize=figsize)
        axes = fig.add_subplot(projection='3d')

        ani = animation.FuncAnimation(
            fig=fig, 
            func=lambda i, axes: self._update(i, axes), 
            frames=len(self.sim.joint_space_position_history), 
            interval=dt*1000, 
            fargs=(axes,)
        )

        if save_path is None: 
            plt.show()
        else:
            with tqdm.tqdm(total=len(self.sim.joint_space_position_history)) as bar:
                ani.save(save_path, progress_callback=lambda i, n: bar.update())
                plt.clf()
                plt.cla()
                plt.close()