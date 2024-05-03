import typing
import pandas as pd

from pyboatsim.kinematics.topology import Topology

class State:
    """
    A class representing data that can be stored for a given topology tree.
    Basically just wraps a dictionary where keys are body names of a topology,
    or "time".
    Could represent velocities, forces, anything that we might want to track
    over time.
    """

    def __init__(self, topology:Topology):
        self.data = {"time": []}
        for body_name in topology.get_ordered_body_list():
            self.data[body_name] = []

    def __getitem__(self, body_names:typing.List[str]=None) -> typing.Dict[str, float]:
        """
        Returns (a subset of) the state.
        """
        return {body_name: self.data[body_name] for body_name in body_names}
            
    def __setitem__(self, body_name, value):
        self.data[body_name] = value

    def bodies(self):
        return self.data.keys()
