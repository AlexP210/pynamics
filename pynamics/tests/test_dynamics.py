import os
import unittest
import warnings

import numpy as np
import matplotlib.pyplot as plt
import trimesh

import pynamics.kinematics.topology as topo
import pynamics.kinematics.joint as joint
import pynamics.dynamics as dynamics
import pynamics.math.linalg as linalg
from pynamics.sim import Sim
import pynamics.constants as const
from pynamics.visualizer import Visualizer


class TestDynamics(unittest.TestCase):

    def save_plot_artifact(self, time, value, expected_value, value_name, test_name):
        if os.getenv("GITHUB_ACTIONS"):
            return

        fig, ax = plt.subplots(1, 2)

        ax[0].scatter(time, value, s=5, label=f"Simulated {value_name}")
        ax[0].scatter(time, expected_value, s=5, label=f"Expected {value_name}")
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel(value_name)
        ax[0].legend()

        ax[1].scatter(
            time, value - expected_value, c="k", s=5, label=f"{value_name} Error"
        )
        ax[1].hlines(
            y=[-const.EPSILON, const.EPSILON],
            xmin=time[0],
            xmax=time[-1],
            colors="r",
            linestyles="--",
            label="Tolerance",
        )
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel(value_name)
        ax[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(const.HOME, "tests", f"{test_name}.png"))
        plt.clf()
        plt.cla()
        plt.close()
    
    def make_cube_topology(self, cube_joint:joint.Joint):
        cube = topo.Body(
            mass_properties_model=trimesh.load(
                file_obj=os.path.join(const.HOME, "models", "common", "Cube.obj"),
                file_type="obj",
                force="mesh",
            ),
            density=990,  # A little bit less dense than water
        )
        corner = topo.Frame(translation=np.matrix([1, 1, 1]).T)
        cube.add_frame(corner, "Corner")

        topology = topo.Topology()
        topology.add_connection(
            parent_body_name="World",
            parent_frame_name="Identity",
            child_body=cube.copy(),
            child_body_name="Cube",
            joint=cube_joint,
        )

        visualizer = Visualizer(
            topology=topology,
            visualization_models={
                ("Cube", "Identity"): trimesh.load(
                    file_obj=os.path.join(const.HOME, "models", "common", "Cube.obj"),
                    file_type="obj",
                    force="mesh",
                )
            },
        )

        return topology, visualizer

    def test_buoyancy(self):
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

        topology, visualizer = self.make_cube_topology(joint.FreeJoint())
        topology.joints["Cube"].set_configuration(np.matrix([1, 0, 0, 0, 0, 0, -500]).T)
        sim = Sim(
            topology=topology,
            body_dynamics={
                "buoyancy": dynamics.Buoyancy(
                    buoyancy_models={
                        "Cube": trimesh.load(
                            file_obj=os.path.join(
                                const.HOME, "models", "common", "Cube.obj"
                            ),
                            file_type="obj",
                            force="mesh",
                        )
                    },
                    fluid_density=1000,
                )
            },
        )
        sim.simulate(delta_t=5, dt=0.05)
        if not os.getenv("GITHUB_ACTIONS"):
            visualizer.add_sim_data(sim)
            visualizer.animate(
                save_path=os.path.join(const.HOME, "tests", "test_buoyancy.mp4")
            )

        t = np.array(sim.data["Time"])
        x = sim.data["Joints"]["Cube"]["Position 6"]
        f = 9.81 * 1000 * (2**3)
        a = f / sim.topology.bodies["Cube"].mass
        x_expected = -500 + 0.5 * (a * t**2)
        self.save_plot_artifact(
            time=t,
            value=x,
            expected_value=x_expected,
            value_name="Z Position",
            test_name="test_buoyancy",
        )
        self.assertTrue(
            (abs(x - x_expected) < const.EPSILON).all(),
            msg="Buoyancy simulates correctly.",
        )

    def test_gravity(self):
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        topology, visualizer = self.make_cube_topology(joint.FreeJoint())
        # topology.joints["Cube"].set_configuration(np.matrix([1/np.sqrt(4),1/np.sqrt(4),1/np.sqrt(4),1/np.sqrt(4),5,0,0]).T)
        topology.joints["Cube"].set_configuration(np.matrix([1, 0, 0, 0, 5, 0, 0]).T)
        sim = Sim(
            topology=topology,
            body_dynamics={"gravity": dynamics.Gravity(g=-9.81, body_names="Cube")},
        )
        sim.simulate(delta_t=5, dt=0.05)
        if not os.getenv("GITHUB_ACTIONS"):
            visualizer.add_sim_data(sim)
            visualizer.animate(
                save_path=os.path.join(const.HOME, "tests", "test_gravity.mp4")
            )

        t = np.array(sim.data["Time"])
        a = -9.81
        x_expected = 0.5 * a * (t**2)
        x = sim.data["Joints"]["Cube"]["Position 6"]

        self.save_plot_artifact(
            time=t,
            value=x,
            expected_value=x_expected,
            value_name="Z Position",
            test_name="test_gravity",
        )

        self.assertTrue(
            (abs(x - x_expected) < const.EPSILON).all(),
            msg="Gravity simulates correctly on a free body.",
        )

    def test_drag(self):
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        topology, visualizer = self.make_cube_topology(joint.FreeJoint())
        topology.joints["Cube"].set_configuration_d(np.matrix([0, 0, 0, 0.1, 0, 0]).T)
        topology.joints["Cube"].set_configuration(np.matrix([1, 0, 0, 0, 0, 0, -10]).T)
        sim = Sim(
            topology=topology,
            body_dynamics={
                "drag": dynamics.QuadraticDrag(
                    drag_models={
                        "Cube": trimesh.load(
                            file_obj=os.path.join(
                                const.HOME, "models", "common", "Cube.obj"
                            ),
                            file_type="obj",
                            force="mesh",
                        )
                    },
                    drag_coefficient=1,
                )
            },
        )
        sim.simulate(delta_t=5, dt=0.05)
        if not os.getenv("GITHUB_ACTIONS"):
            visualizer.add_sim_data(sim)
            visualizer.animate(save_path=os.path.join(const.HOME, "tests", "test_drag.mp4"))

        t = np.array(sim.data["Time"])
        # v_d = (-0.5 * density * C_d * A) / m * v^2
        # => v_d(t) = 1 / [(0.5*density*C_d*A)*t + c_1]
        # => c_1 = 1/v_d(0)
        m = sim.topology.bodies["Cube"].mass
        v_expected = 1 / (0.5 * 1000 * 1 * (2**2) / m * t + 1 / 0.1)
        v = sim.data["Joints"]["Cube"]["Velocity 3"]

        self.save_plot_artifact(
            time=t,
            value=v,
            expected_value=v_expected,
            value_name="X Velocity",
            test_name="test_drag",
        )

        self.assertTrue(
            (abs(v - v_expected) < const.EPSILON).all(),
            msg="Drag simulates correctly on a free body.",
        )

    def test_spring(self):
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        topology, visualizer = self.make_cube_topology(joint.TranslationJoint())
        topology.joints["Cube"].set_configuration_d(np.matrix([0, 0, -0.1]).T)
        sim = Sim(
            topology=topology,
            body_dynamics={
                "spring": dynamics.Spring(
                    body1="World",
                    frame1="Identity",
                    body2="Cube",
                    frame2="Identity",
                    stiffness=topology.bodies["Cube"].mass,
                )
            },
        )
        sim.simulate(delta_t=5, dt=0.01)
        visualizer.add_sim_data(sim)
        if not os.getenv("GITHUB_ACTIONS"):
            visualizer.animate(
                save_path=os.path.join(const.HOME, "tests", "test_spring.mp4")
            )

        t = np.array(sim.data["Time"])
        # x_dd = -(k/m)*(|x|-l)
        # x(t) = A*sin(sqrt(k/m)*t) + B*cos(sqrt(k/m)*t) + l
        # x_d(t) = A*sqrt(k/m)*cos(sqrt(k/m)*t) - B*sqrt(k/m)*sin(sqrt(k/m)*t)
        # x_dd(t) = -A*(k/m)*sin(sqrt(k/m)*t) - B*(k/m)*cos(sqrt(k/m)*t)
        # x_dd(t) = -(k/m) * ( A*sin(sqrt(k/m)*t) + B*cos(sqrt(k/m)*t) ) = -(k/m)*x(t) - l
        # => x(0) = B + l => B = x(0) - l
        # => x_d(0) = A*sqrt(k/m) => A = x_d(0)/omega * x_d(0)
        m = sim.topology.bodies["Cube"].mass
        k = m
        omega = np.sqrt(k / m)
        x_expected = (-0.1 / omega) * np.sin(omega * t)
        x = sim.data["Joints"]["Cube"]["Position 2"]

        self.save_plot_artifact(
            time=t,
            value=x,
            expected_value=x_expected,
            value_name="Z Position",
            test_name="test_spring",
        )

        self.assertTrue(
            (abs(x - x_expected) < const.EPSILON).all(),
            msg="Spring simulates correctly on a free body.",
        )

    def test_joint_damping(self):
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        topology, visualizer = self.make_cube_topology(joint.RevoluteJoint(0))
        topology.joints["Cube"].set_configuration_d(np.matrix([0.1,]).T)
        sim = Sim(
            topology=topology,
            joint_dynamics={
                "damping": dynamics.JointDamping(
                    damping_factor=0.5,
                    joint_names=["Cube"]
                )
            },
        )
        sim.simulate(delta_t=5, dt=0.01)
        visualizer.add_sim_data(sim)
        if not os.getenv("GITHUB_ACTIONS"):
            visualizer.animate(
                save_path=os.path.join(const.HOME, "tests", "test_joint_damping.mp4")
            )

        t = np.array(sim.data["Time"])
        # x_dd = -(k/m)*x_d
        # v_d = -(k/m)*v
        # (1/v)*dv = -(k/m)*dt
        # ln(v) + C = -(k/m)*t
        # v(t) = Ae^(-(k/m)*t)
        # => v(0) = A
        # x_d(t) = v(0)*e^(-kt/m)
        # x(t) = (-v(0)m/k)*e^(-k*t/m) + B
        # => B = x(0) + v(0)m/k
        # x(t) = (-v(0)m/k)*e^(-k*t/m) + x(0) + v(0)m/k
        x_hat = np.matrix([1,0,0]).T
        I = (x_hat.T @ topology.bodies["Cube"].inertia_matrix @ x_hat)[0,0]
        x_expected = (0.1 * I / 0.5) * (1 - np.exp(-0.5 * t / I))
        x = sim.data["Joints"]["Cube"]["Position 0"]

        self.save_plot_artifact(
            time=t,
            value=x,
            expected_value=x_expected,
            value_name="Joint Position",
            test_name="test_joint_damping",
        )

        self.assertTrue(
            (abs(x - x_expected) < const.EPSILON).all(),
            msg="Joint damping simulates correctly on a free body.",
        )

    def test_fixed_joint(self):
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        topology, visualizer = self.make_cube_topology(joint.FixedJoint())
        sim = Sim(
            topology=topology,
            body_dynamics={
                "gravity": dynamics.Gravity(
                    g=-9.81,
                    direction=np.matrix([0,0,1]).T,
                    body_names=["Cube"]
                )
            },
        )
        sim.simulate(delta_t=5, dt=0.01)
        visualizer.add_sim_data(sim)
        if not os.getenv("GITHUB_ACTIONS"):
            visualizer.animate(
                save_path=os.path.join(const.HOME, "tests", "test_joint_damping.mp4")
            )

        t = np.array(sim.data["Time"])
        # x_dd = -(k/m)*x_d
        # v_d = -(k/m)*v
        # (1/v)*dv = -(k/m)*dt
        # ln(v) + C = -(k/m)*t
        # v(t) = Ae^(-(k/m)*t)
        # => v(0) = A
        # x_d(t) = v(0)*e^(-kt/m)
        # x(t) = (-v(0)m/k)*e^(-k*t/m) + B
        # => B = x(0) + v(0)m/k
        # x(t) = (-v(0)m/k)*e^(-k*t/m) + x(0) + v(0)m/k
        self.assertTrue(
            sim.data["Joints"]["Cube"] == {}
        )

    def test_revolute_motor(self):
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        topology, visualizer = self.make_cube_topology(joint.RevoluteJoint(0))
        sim = Sim(
            topology=topology,
            joint_dynamics={
                "damping": dynamics.RevoluteDCMotor(
                    joint_name="Cube",
                    electromotive_constant=10,
                    resistance=0.1,
                    inductance=0.1,
                    voltage=100
                )
            },
        )
        sim.simulate(delta_t=5, dt=0.01)
        visualizer.add_sim_data(sim)
        if not os.getenv("GITHUB_ACTIONS"):
            visualizer.animate(
                save_path=os.path.join(const.HOME, "tests", "test_revolute_motor.mp4")
            )

        ts = np.array(sim.data["Time"])

        # https://ctms.engin.umich.edu/CTMS/index.php?example=MotorSpeed&section=SystemModeling
        # x = [w, i].T
        # A = [-b/J,  K/J 
        #      -K/L, -R/L]
        # B = [0, V/L].T
        # x_d = Ax + B
        # => x(t) = e^(At)*C + x_p
        # => x_p = e^(At)*int(0, t, e^(-At)*B)
        # => x_p = e^(At) * (-inv(A)*e^(-At)*B)
        # x(t) = e^(At)*C +  e^(At) * (-inv(A)*e^(-At)*B)
        # => x(0) = 0
        # => C = inv(A)*B
        # x(t) = e^(At)*inv(A)*B +  e^(At) * (-inv(A)*e^(-At)*B)
        x_hat = np.matrix([1,0,0]).T
        J = (x_hat.T @ topology.bodies["Cube"].inertia_matrix @ x_hat)[0,0] # inertia
        b = 0 # viscous damping
        K = 10 # Electromotive constant
        L = 0.1 #inductance
        R = 0.1 # resistance
        V = 100 # Voltage
        A = np.matrix([
            [-b/J, K/J],
            [-K/L, -R/L]
        ])
        inv_A = np.linalg.inv(A)
        B = np.matrix([0, V/L]).T
        x_expected = np.zeros(ts.size)

        for t_idx in range(len(ts)):
            t = ts[t_idx]
            exp_At = linalg.matrix_exponential(A*t)
            neg_exp_At = linalg.matrix_exponential(-A*t)

            x_expected[t_idx] = (exp_At@inv_A@(np.eye(*A.shape) - neg_exp_At)@B)[0,0]
        x = sim.data["Joints"]["Cube"]["Velocity 0"]

        self.save_plot_artifact(
            time=ts,
            value=x,
            expected_value=x_expected,
            value_name="Joint Position",
            test_name="test_revolute_motor",
        )

        self.assertTrue(
            (abs(x - x_expected) < 0.005).all(),
            msg="RevoluteDCMotor simulates correctly on a free body.",
        )


    def test_constant_joint_force(self):
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        topology, visualizer = self.make_cube_topology(joint.FreeJoint())
        sim = Sim(
            topology=topology,
            joint_dynamics={
                "constant joint force": dynamics.ConstantJointForce(
                    force=np.matrix([0,0,0, 1,0,0]).T
                ),
            },
        )
        sim.simulate(delta_t=5, dt=0.01)
        visualizer.add_sim_data(sim)
        if not os.getenv("GITHUB_ACTIONS"):
            visualizer.animate(
                save_path=os.path.join(const.HOME, "tests", "test_constant_joint_force.mp4")
            )

        ts = np.array(sim.data["Time"])

        x_expected = 0.5*(1/topology.bodies["Cube"].mass) * ts**2
        x = sim.data["Joints"]["Cube"]["Position 4"]

        self.save_plot_artifact(
            time=ts,
            value=x,
            expected_value=x_expected,
            value_name="Joint Position",
            test_name="test_constant_joint_force",
        )

        self.assertTrue(
            (abs(x - x_expected) < const.EPSILON).all(),
            msg="ConstantJointForce simulates correctly on a free body.",
        )

    def test_constant_body_force(self):
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
        topology, visualizer = self.make_cube_topology(joint.FreeJoint())
        sim = Sim(
            topology=topology,
            body_dynamics={
                "constant body force": dynamics.ConstantBodyForce(
                    force=np.matrix([1,0,0]).T,
                ),
            },
        )
        sim.simulate(delta_t=5, dt=0.01)
        visualizer.add_sim_data(sim)
        if not os.getenv("GITHUB_ACTIONS"):
            visualizer.animate(
                save_path=os.path.join(const.HOME, "tests", "test_constant_body_force.mp4")
            )

        ts = np.array(sim.data["Time"])

        x_expected = 0.5*(1/topology.bodies["Cube"].mass) * ts**2
        x = sim.data["Joints"]["Cube"]["Position 4"]

        self.save_plot_artifact(
            time=ts,
            value=x,
            expected_value=x_expected,
            value_name="Joint Position",
            test_name="test_constant_body_force",
        )

        self.assertTrue(
            (abs(x - x_expected) < const.EPSILON).all(),
            msg="ConstantBodyForce simulates correctly on a free body.",
        )
