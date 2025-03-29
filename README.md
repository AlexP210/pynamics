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
For usage instructions, along with detailed API documentation, see the [documentation](https://alexp210.github.io/pynamics/).

Also see the available [usage examples](pynamics/example).