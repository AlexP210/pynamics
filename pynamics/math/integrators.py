"""Module containing definition of various classes for performing integration."""
import abc
import quaternion
import numpy as np
import typing


class Integrator(abc.ABC):
    """Base class for integrators"""
    @abc.abstractmethod
    def __init__(self):
        pass

    def set_initial_condition(self, initial_condition:np.matrix):
        """
        Initialize the state to integrate from.

        Args:
            initial_condition (np.matrix): The initial state.
        """
        self.state = initial_condition

    @abc.abstractmethod
    def integrate(self, dt:float, derivative_function:typing.Callable[[np.matrix], np.matrix]) -> np.matrix:
        """
        Step the integrator.

        Args:
            dt (float): Time integral to integrate over.
            derivative_function (typing.Callable[[np.matrix], np.matrix]): A function that operates
            on a state to produce a derivative.

        Returns:
            np.matrix: The updated state.
            np.matrix: The derivative evaluation
        """
        pass

class ForwardEuler(Integrator):
    """Class for performing Forward Euler integration on quaternions."""
    def __init__(self):
        pass

    def set_initial_condition(self, initial_condition):
        super().set_initial_condition(initial_condition)

    def integrate(self, dt, derivative_function):
        derivative = derivative_function(self.state)
        self.state += derivative * dt
        return self.state, derivative
    
class RungeKutta4(Integrator):
    """Class for performing Runge-Kutta 4 Integration"""
    def __init__(self):
        pass

    def set_initial_condition(self, initial_condition):
        super().set_initial_condition(initial_condition)

    def integrate(self, dt, derivative_function):
        k1 = derivative_function(self.state)
        k2 = derivative_function(self.state+dt*k1/2)
        k3 = derivative_function(self.state+dt*k2/2)
        k4 = derivative_function(self.state+dt*k3)

        self.state += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        return self.state, k1