import os
import unittest
import warnings

import numpy as np
import trimesh

import pyboatsim.kinematics.topology as topo
import pyboatsim.kinematics.joint as joint
import pyboatsim.dynamics as dynamics
from pyboatsim.boatsim import Sim
import pyboatsim.constants as const


class TestDynamics(unittest.TestCase):

    def make_topology(self):
        cube = topo.Body(
            mass_properties_model=trimesh.load(
                file_obj=os.path.join(const.HOME, "models", "common", "Cube.obj"),
                file_type="obj",
                force="mesh"
            ),
            density = 990 # A little bit less dense than water
        )
        corner = topo.Frame(
            translation=np.matrix([1,1,1]).T
        )
        cube.add_frame(corner, "Corner")
    
        topology = topo.Topology()
        topology.add_connection(
            parent_body_name=f"World",
            parent_frame_name="Identity",
            child_body=cube.copy(),
            child_body_name=f"Cube",
            joint=joint.FreeJoint()
        )
        return topology

    def test_buoyancy(self):
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning) 
        topology = self.make_topology()
        topology.joints["Cube"].set_configuration(np.matrix([0,0,0,0,0,-10]).T)
        sim = Sim(
            topology=topology,
            body_dynamics={
                "buoyancy": dynamics.Buoyancy(
                    buoyancy_models={
                        "Cube": trimesh.load(
                            file_obj=os.path.join(const.HOME, "models", "common", "Cube.obj"),
                            file_type="obj",
                            force="mesh")
                    },
                    fluid_density=1000
                )
            }
        )
        sim.simulate(delta_t=0.05, dt=0.05)
        velocity = np.matrix([
            sim.data_history[f"Cube / Velocity {i}"][-1]
            for i in range(3, 6)
        ]).T 
        acceleration = velocity/0.05
        force = topology.bodies["Cube"].mass * acceleration
        force_expected = np.matrix([0, 0, 2**3 * 1000 * 9.81]).T
        self.assertTrue(
            (abs(force - force_expected) < const.EPSILON).all(),
            msg="Buoyancy calculates expected force."
        )

    def test_gravity(self):
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning) 
        topology = self.make_topology()
        topology.joints["Cube"].set_configuration(np.matrix([0,0,0,5,0,0]).T)
        sim = Sim(
            topology=topology,
            body_dynamics={
                "gravity": dynamics.Gravity(
                    g = -9.81,
                    direction=2,
                    body_names="Cube"
                )
            }
        )
        sim.simulate(delta_t=0.05, dt=0.05)
        velocity = np.matrix([
            sim.data_history[f"Cube / Velocity {i}"][-1]
            for i in range(3, 6)
        ]).T 
        acceleration = velocity/0.05
        force = topology.bodies["Cube"].mass * acceleration
        force_expected = np.matrix([0, 0, -9.81 * 990*(2**3)]).T
        self.assertTrue(
            (abs(force - force_expected) < const.EPSILON).all(),
            msg="Buoyancy calculates expected force."
        )
