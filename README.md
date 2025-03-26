# Pynamics

[![tests](https://github.com/AlexP210/pynamics/actions/workflows/tests.yml/badge.svg)](https://github.com/AlexP210/pynamics/actions/workflows/tests.yml)
![coverage](.coverage/coverage-badge.svg)

Pynamics is a simple multi-rigid-body dynamics simulator writen in pure python, following the formulation in *Rigid Body Dynamics Algorithms* by Roy Featherstone (2008).

With Pynamics, you can quickly configure kinematic tree topologies, and define custom dynamics modules to add arbitrary physics sources to your simulation.

## Installation
Currently, Pynamics only supports developer-level installation. To install Pynamics in this way:
```bash
git clone https://github.com/AlexP210/pynamics
cd pynamics
pip install -r requirements.txt
```

## Usage
Pynamics uses the following workflow for configuring simulations:

1.  Define the `Body` objects representing the rigid bodies in the simulation
1.  Create a `Topology` object, and assemble the `Body` objects using `Joint` objects.
1.  Initialize the `Topology` with the desired initial configuration.
1.  Create a `Sim` object, and add the `Topology` and any combination `BodyDynamics` or `JointDynamics` objects, to provide the external forces on the `Topology`
1.  Call `Sim.simulate()` to run the simulation

Out of the box, Pynamics provides a variety of joint implementations and dynamics modules. See `pynamics/kinematics/joint.py` and `pynamics/dynamics/*.py` for implementations.

For example usage, see `pynamics/example`.