import os
import unittest
import warnings

import numpy as np
import trimesh

import pyboatsim.kinematics.topology as topo
import pyboatsim.constants as const


class TestTopology(unittest.TestCase):
    body = topo.Body()
    body.add_frame(
        frame = topo.Frame(
            translation=np.matrix([1,0,0]).T,
            rotation=topo.Frame.get_rotation_matrix(
                angle=np.pi/2,
                axis=np.matrix([0,0,1]).T
            )
        ),
        frame_name="Attachment Point"
    )

    def make_body(self):
        cube = topo.Body(
            mass_properties_model=trimesh.load(
                file_obj=os.path.join(const.HOME, "models", "common", "Cube.obj"),
                file_type="obj",
                force="mesh"
            ),
            density = 10
        )
        corner = topo.Frame(
            translation=np.matrix([1,1,1]).T
        )
        cube.add_frame(corner, "Corner")

        return cube


    def make_topology(self):
        topology = topo.Topology()
        topology.add_connection(
            parent_body_name=f"World",
            parent_frame_name="Identity",
            child_body=self.body.copy(),
            child_body_name=f"Body0",
            joint=topo.RevoluteJoint(axis=0)
        )
        for i in range(1, 4):
            topology.add_connection(
                parent_body_name=f"Body{i-1}",
                parent_frame_name="Attachment Point",
                child_body=self.body.copy(),
                child_body_name=f"Body{i}",
                joint=topo.RevoluteJoint(axis=0)
            )
        return topology

    def test_topology_definition(self):
        """
        Test that a Topology can be instantiated
        """
        topology = self.make_topology()
        self.assertEqual(
            first=len(topology.bodies), 
            second=5, 
            msg="Topology contains all of the added bodies"
        )
        self.assertEqual(
            first=len(topology.joints), 
            second=5, 
            msg="Topology contains all of the added joints"
        )
    
    def test_topology_articulation(self):
        """
        Test that a topology can be articulated.
        """
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning) 
        
        topology = self.make_topology()
        T = topology.get_transform(
            from_body_name="Body0", from_frame_name="Identity",
            to_body_name="Body3", to_frame_name="Attachment Point"
        )
        r = T[:3, 3]
        C = T[:3,:3]
        self.assertTrue(
            (abs(r - np.matrix(np.zeros(shape=(3,1)))) < const.EPSILON).all(),
            msg="Last frame in circular topology matches with first in position."
        )
        self.assertTrue(
            (abs(C - np.matrix(np.eye(N=3,M=3))) < const.EPSILON).all(),
            msg="Last frame in circular topology matches with first in orientation."
        )
        self.assertTrue(
            (T[3,:4] == np.matrix([0, 0, 0, 1])).all(),
            msg="Affine transformation matrix has correct form."
        )

        topology.joints["Body2"].set_configuration(np.matrix([[np.pi/2]]))
        T = topology.get_transform(
            from_body_name="Body0", from_frame_name="Identity",
            to_body_name="Body3", to_frame_name="Attachment Point"
        )
        r = T[:3, 3]
        C = T[:3,:3]
        r_expected = np.matrix([0, 1, 1]).T
        self.assertTrue(
            (abs(r - r_expected) < const.EPSILON).all(),
            msg="Last frame in articulated topology has expected position."
        )
        C_expected = topo.Frame.get_rotation_matrix(angle=-np.pi/2,axis=np.matrix([1,0,0]).T)
        self.assertTrue(
            (abs(C - C_expected) < const.EPSILON).all(),
            msg="Last frame in articulated topology has expected orientation."
        )
        self.assertTrue(
            (T[3,:4] == np.matrix([0, 0, 0, 1])).all(),
            msg="Affine transformation matrix has correct form."
        )

    def test_topology_forces(self):
        """
        Test that topology successfully applies forces
        """
        topology = self.make_topology()
        X = topology.get_Xstar(
            from_body_name="Body0", from_frame_name="Attachment Point",
            to_body_name="Body0", to_frame_name="Identity"
        )
        wrench = np.matrix([0, 1, 0, 1, 0, 0]).T # Pull and push down
        transformed_wrench = X@wrench
        transformed_wrench_expected = np.matrix([-1, 0, 1, 0, 1, 0]).T
        self.assertTrue(
            (abs(transformed_wrench - transformed_wrench_expected) < const.EPSILON).all(),
            msg="Transformed wrench is correctly calculated."
        )

    def test_topology_velocities(self):
        """
        Test that topology successfully transforms velocities
        """
        topology = self.make_topology()
        X = topology.get_X(
            from_body_name="Body0", from_frame_name="Attachment Point",
            to_body_name="Body0", to_frame_name="Identity"
        )
        velocity = np.matrix([0, 0, 1, 1, 0, 0]).T # Pull and push down
        transformed_velocity = X@velocity
        transformed_velocity_expected = np.matrix([0, 0, 1, 0, 0, 0]).T
        self.assertTrue(
            (abs(transformed_velocity - transformed_velocity_expected) < const.EPSILON).all(),
            msg="Transformed velocity is correctly calculated."
        )


    def test_body_massproperties(self):
        """
        Test that a body can be instantiated and calculated mass properties are
        correct.
        """
        warnings.filterwarnings("ignore", category=PendingDeprecationWarning) 

        cube = self.make_body()

        m = cube.mass
        m_expected = 10*2**3
        self.assertTrue(
            abs(m - m_expected) < const.EPSILON,
            msg="Mass is correctly calculated."
        )

        c = cube.center_of_mass
        c_expected = np.matrix(np.zeros((1,3)))
        self.assertTrue(
            (abs(c - c_expected) < const.EPSILON).all(),
            msg="Center of mass is correctly calculated."
        )

        J = cube.inertia_matrix
        J_expected = (10*2**3)*(2**2/6) * np.matrix(np.eye(3,3))
        self.assertTrue(
            (abs(J - J_expected) < const.EPSILON).all(),
            msg="Inertia tensor is correctly calculated."
        )

    def test_topology_velocities(self):
        """
        Test that topology accurately calculates body velocities given joint velocity
        """
        topology = self.make_topology()
        topology.joints["Body1"].set_configuration_d(np.matrix([[1]]))
        topology.update_body_velocities()
        velocity = topology.bodies["Body3"].get_velocity()
        velocity_expected = np.matrix([-1, 0, 0, 0, 0, 1]).T
        self.assertTrue(
            (abs(velocity - velocity_expected) < const.EPSILON).all(),
            msg="Body velocity is correctly calculated."
        )

    def test_topology_accelerations(self):
        """
        Test that topology accurately calculates body accelerations given joint acceleration
        """
        topology = self.make_topology()
        q_dd = {"Body1": np.matrix([[1]])}
        topology.joints["Body1"].set_configuration_d(np.matrix([[1]]))
        topology.update_body_velocities()
        
        body_accelerations = topology.calculate_body_accelerations(q_dd)
        acceleration = body_accelerations["Body3"]
        acceleration_expected = np.matrix([-1, 0, 0, 0, 0, 1]).T
        self.assertTrue(
            (abs(acceleration - acceleration_expected) < const.EPSILON).all(),
            msg="Body acceleration is correctly calculated."
        )
