import trimesh
import numpy as np
from pyboatsim.boatsim import BoAtSim
from pyboatsim.topology import Topology
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
            visualization_models:typing.Dict,
            boatsim:BoAtSim=None, 
        ):
        self.topology=topology
        self.visualization_models = visualization_models
        if boatsim is not None: self.add_sim_data(boatsim=boatsim)

    def add_sim_data(self, boatsim:BoAtSim):
        self.boatsim = boatsim
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

        # Get the state for this time
        state = self.data.iloc[step_idx]

        # Array representation of position & rotation of root body inertial frame
        theta = np.array([state[f"theta_{axis}__boat"] for axis in AXES])
        r = np.array([state[f"r_{axis}__boat"] for axis in AXES])
        c = np.array([state[f"c_{axis}__boat"] for axis in AXES])
        com__worldframe = r+c

        axes.clear()
        axes.set_xlim(r[0]-2, r[0]+2)
        axes.set_ylim(r[1]-2, r[1]+2)
        axes.set_zlim(r[2]-2, r[2]+2)
        
        # Place the water
        xx, yy = np.meshgrid((r[0]-2, r[0]+2), (r[1]-2, r[1]+2))
        z = state["r_z__water"]*np.ones(xx.shape)
        water = axes.plot_surface(xx, yy, z, color="b", alpha=0.1, zorder=1)

        # Get transformation from World frame to topology com frame
        T_root_com = trimesh.transformations.translation_matrix(direction=c)
        T_com_root = trimesh.transformations.translation_matrix(direction=-c)
        T_world_root = trimesh.transformations.translation_matrix(direction=r)
        if np.linalg.norm(theta) <= EPSILON:
            C_world_com = np.eye(4)
        else:
            C_world_com = trimesh.transformations.rotation_matrix(
                direction=theta/np.linalg.norm(theta),
                angle=np.linalg.norm(theta)
            )
        transformation_matrix = trimesh.transformations.concatenate_matrices(
                T_root_com,
                T_world_root,
                C_world_com,
                T_com_root
        )
        
        # Transform the mesh
        # Transform it to the CM frame
        artists = []
        for (body_name, frame_name), visualization_model in self.visualization_models.items():
            visualization_model_temp = visualization_model.copy()
            visualization_model_temp.apply_transform(
                transformation_matrix@np.array(self.boatsim.topology.get_transform(
                    from_body_name=self.boatsim.topology.root_body_name,
                    from_frame_name="Identity",
                    to_body_name=body_name,
                    to_frame_name=frame_name
                ))
            )
            # Prepare the axes and plot
            vertices = visualization_model_temp.vertices
            T = visualization_model_temp.faces

            artist = axes.plot_trisurf(
                vertices[:,0], vertices[:,1], vertices[:,2], 
                triangles = T, edgecolor=[[0,0,0]], linewidth=1.0, 
                alpha=0.5, shade=True, color="k",zorder=1
            )
            artists.append(artist)

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
                tau__worldframe_m = C_world_com[:3,:3]@tau__comframe_m
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
        return water, *artists

    def view(self, save_path:str=None, verbose:bool=True, figsize=(12,10)):

        fig = plt.figure(figsize=figsize)
        axes = fig.add_subplot(projection='3d')

        lower_bound = np.zeros(shape=(3,))
        upper_bound = np.zeros(shape=(3,))
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

        center = 0.5 * (upper_bound + lower_bound)
        max_extent = 0
        for i in range(3):
            max_extent = max(max_extent, 0.5*(upper_bound[i] - lower_bound[i]))
        max_extent *= 1.1
        axes.set_xlim(center[0] - max_extent, center[0] + max_extent)
        axes.set_ylim(center[1] - max_extent, center[1] + max_extent)
        axes.set_zlim(center[2] - max_extent, center[2] + max_extent)

        if save_path is None: 
            plt.show()
        else:
            plt.savefig(save_path, dpi=300)
            plt.clf()
            plt.cla()
            plt.close()

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

if __name__ == "__main__":

    from pyboatsim.topology import Body, Frame, Articulation

    body = Body(
        mass=1,
        center_of_mass=np.matrix([0,0.0,0.0]).T,
        inertia_matrix=np.matrix([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
    )
    base_mounting_point = Frame(
        translation=np.matrix([2.0,0.0,0.0]).T, 
        rotation=Frame.get_rotation_matrix(0.0, np.matrix([0.0,0.0,0.0]).T)
    )
    short_end = Frame(
        translation=np.matrix([0.25,0.1,0.0]).T, 
        rotation=Frame.get_rotation_matrix(0.0, np.matrix([0.0,0.0,0.0]).T)
    )
    long_end = Frame(
        translation=np.matrix([1.0,0.1,0.0]).T, 
        rotation=Frame.get_rotation_matrix(0.0, np.matrix([0.0,0.0,0.0]).T)
    )
    long_end_for_yaw = Frame(
        translation=np.matrix([1.0,0.0,0.1]).T, 
        rotation=Frame.get_rotation_matrix(0.0, np.matrix([0.0,0.0,0.0]).T)
    )

    base = body.copy()
    roll_body = body.copy()
    pitch_body_1 = body.copy()
    pitch_body_2 = body.copy()
    yaw_body = body.copy()
    
    base.add_frame(base_mounting_point, "Base to Roll Body")
    roll_body.add_frame(short_end, "Roll Body to Pitch Body 1")
    pitch_body_1.add_frame(long_end, "Pitch Body 1 to Pitch Body 2")
    pitch_body_2.add_frame(long_end_for_yaw, "Pitch Body 2 to Yaw Body")
    yaw_body.add_frame(short_end, "End Effector")


    robot = Topology()
    robot.add_connection("World", "Identity", base, "Base Body")
    robot.add_connection(
        "Base Body", "Base to Roll Body", roll_body, "Roll Body",
        constraints=Articulation.ROTATE_X)
    robot.add_connection(
        "Roll Body", "Roll Body to Pitch Body 1", pitch_body_1, "Pitch Body 1",
        constraints=Articulation.ROTATE_Y)
    robot.add_connection(
        "Pitch Body 1", "Pitch Body 1 to Pitch Body 2", pitch_body_2, "Pitch Body 2",
        constraints=Articulation.ROTATE_Y)
    robot.add_connection(
        "Pitch Body 2", "Pitch Body 2 to Yaw Body", yaw_body, "Yaw Body",
        constraints=Articulation.ROTATE_Z)
    
    # robot.set_articulation("Roll Body", np.array([0,0,0,np.pi/2,0,0]))
    # robot.set_articulation("Pitch Body 1", np.array([0,0,0,0,np.pi/4,0]))
    # robot.set_articulation("Pitch Body 2", np.array([0,0,0,0,np.pi/2,0]))
    robot.set_articulation("Yaw Body", np.array([0,0,0,0,0,np.pi/4]))


    vis = Visualizer(
        topology=robot,
        visualization_models={
            ("Base Body", "Identity"): trimesh.load(
                file_obj="/home/alex/Projects/PyBoAtSim/models/link/Base.obj", 
                file_type="obj", 
                force="mesh"),
            ("Roll Body", "Identity"): trimesh.load(
                file_obj="/home/alex/Projects/PyBoAtSim/models/link/Link0p25m.obj", 
                file_type="obj", 
                force="mesh"),
            ("Pitch Body 1", "Identity"): trimesh.load(
                file_obj="/home/alex/Projects/PyBoAtSim/models/link/Link1m.obj", 
                file_type="obj", 
                force="mesh"),
            ("Pitch Body 2", "Identity"): trimesh.load(
                file_obj="/home/alex/Projects/PyBoAtSim/models/link/Link1m.obj", 
                file_type="obj", 
                force="mesh"),
            ("Yaw Body", "Identity"): trimesh.load(
                file_obj="/home/alex/Projects/PyBoAtSim/models/link/Link0p25m.obj", 
                file_type="obj", 
                force="mesh"),
        })
    
    vis.view()


