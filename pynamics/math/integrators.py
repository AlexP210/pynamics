"""Module containing definition of various classes for performing integration."""
import abc
import quaternion
import numpy as np


class Integrator(abc.ABC):
    """Base class for integrators"""
    @abc.abstractmethod
    def __init__(self):
        pass

    def initialize_state(self, initial_position_state:np.matrix, initial_velocity_state:np.matrix):
        """
        Initialize the state to integrate from.

        Args:
            initial_position_state (np.matrix): The initial position.
            initial_velocity_state (np.matrix): THe initial velocity.
        """
        self.position_state = initial_position_state
        self.velocity_state = initial_velocity_state

    @abc.abstractmethod
    def step(self, dt:float, acceleration:np.matrix):
        """
        Step the integrator.

        Args:
            dt (float): Timestep to step the integrator by.
            acceleration (np.matrix): Acceleration to apply.
        """
        pass


class VerletIntegrator(Integrator):
    """Class for performing Verlet Integration"""
    def __init__(self):
        self.position_state_history = []
        self.velocity_state_history = []

    def step(self, dt, acceleration):
        if len(self.position_state_history) == 1:
            xn = self.position_state_history[0]
            vn = self.velocity_state_history[0]
            vnplus1 = vn + acceleration * dt
            xnplus1 = xn + vn * dt + 0.5 * acceleration * dt**2
        else:
            xn = self.position_state_history[-1]
            vn = self.velocity_state_history[-1]
            xnminus1 = self.position_state_history[-2]
            xnplus1 = 2 * xn - xnminus1 + acceleration * dt**2
            vnplus1 = 2 * (xnplus1 - xn) / dt - vn

        self.position_state = xnplus1
        self.velocity_state = vnplus1
        self.position_state_history.append(self.position_state)
        self.velocity_state_history.append(self.velocity_state)
        return self.position_state, self.velocity_state

    def initialize_state(self, initial_position_state, initial_velocity_state):
        super().initialize_state(initial_position_state, initial_velocity_state)
        self.position_state_history = [
            initial_position_state,
        ]
        self.velocity_state_history = [
            initial_velocity_state,
        ]


class ForwardEulerQuaternionIntegrator(Integrator):
    """Class for performing Forward Euler integration on quaternions."""
    def __init__(self):
        pass

    def step(self, dt, acceleration):
        vn = self.velocity_state
        vnplus1 = vn + acceleration * dt
        v_avg = 0.5 * (vn + vnplus1)
        delta_rotation = quaternion.from_rotation_vector(v_avg.T * dt)
        current_rotation = quaternion.from_float_array(self.position_state.T)
        xnplus1 = quaternion.as_float_array(delta_rotation * current_rotation).T
        self.position_state = xnplus1
        self.velocity_state = vnplus1
        return self.position_state, self.velocity_state
