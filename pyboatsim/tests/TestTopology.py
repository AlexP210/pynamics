import unittest

import numpy as np

from pyboatsim.kinematics.topology import Frame, Body, Topology

class TestTopology(unittest.TestCase):
    
    body = Body()
    body.add_frame(
        frame = Frame(
            translation=np.matrix([1,0,0]).T,
            rotation=np.matrix([
                [1, 0, 0],
                [0, 1, 1],
                [0, -1, 0]
            ])
        )
    )

    def test_topology_definition(self): pass