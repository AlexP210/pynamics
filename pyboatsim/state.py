import typing
import pandas as pd

class State:
    def __init__(self, state_dictionary: typing.Dict[str, float] = {}):
        self._state_dictionary = state_dictionary


    def load(self, name, state_database: pd.DataFrame) -> None:
        """
        Selects the state with index `name` from `state_database`.
        `state_database` can be a database of initial states or a previous
        sim output.
        """
        # Pandas exports the dataframe as {col: {index: value}} format, even if
        # there's only one index value in the dataframe. We need {col: value} 
        # for the index selected
        state_database_as_dict = state_database.to_dict()
        state_dict = {col: state_database_as_dict[col][name] 
                      for col in state_database_as_dict.keys()}
        self.set(state_dict)

    def set(self, partial_state_dictionary:typing.Dict[str, float]) -> None:
        """
        Update the simulation parameters without having to specify the
        whole `state` dictionary.
        """
        self._state_dictionary.update(partial_state_dictionary)

    def get(self, state_labels:typing.List[str]=None) -> typing.Dict[str, float]:
        """
        Returns (a subset of) the state.
        """
        if state_labels is None: state_labels = self._state_dictionary.keys()
        return {label: self._state_dictionary[label] for label in state_labels}
    
    def copy(self):
        return State(state_dictionary=self.get())
    
    def __getitem__(self, key):
        return self._state_dictionary[key]
    
    def __setitem__(self, key, value):
        self._state_dictionary[key] = value

    def labels(self):
        return self._state_dictionary.keys()


    
