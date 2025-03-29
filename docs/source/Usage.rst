=====
Usage
=====

Pynamics uses the following workflow for configuring simulations:

1. Define a `Topology` object representing your desired kinematic tree.
2. Initialize the `Topology`` with the desired initial configuration.
3. Define a `Sim` object using your `Topology`, along with any desired `BodyDynamics`
    or `JointDynamics` objects for applying forces to your `Topology`.
4. Call `Sim.simulate()` to run your simulation.
5. Analyze your sim data:
    1. Plot data from `Sim.data_history`
    2. Visualize data with `Visualizer`.


---------------------
Defining a `Topology`
---------------------
