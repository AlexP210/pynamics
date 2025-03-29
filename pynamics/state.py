import typing
import pandas as pd

from pynamics.kinematics.topology import Topology

class State:
    """
    A class representing data that can be stored for a given topology tree.
    Basically just wraps a dictionary where keys are body names of a topology,
    or "time".
    Could represent velocities, forces, anything that we might want to track
    over time.
    """

    def __init__(self, topology:Topology):
        self.data = {}
        for body_name in topology.get_ordered_body_list():
            self.data[body_name] = None

    def __getitem__(self, body_name:str) -> typing.Dict[str, float]:
        """
        Returns (a subset of) the state.
        """
        return self.data[body_name]
            
    def __setitem__(self, body_name, value):
        self.data[body_name] = value

    def bodies(self):
        return self.data.keys()
    
    def clear(self):
        self.data = {}
        for body_name in self.data.keys():
            self.data[body_name] = None
