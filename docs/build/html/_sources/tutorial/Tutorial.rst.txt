=========
Tutorial
=========

Pynamics uses the following workflow for configuring simulations:

1. Define a :code:`Topology` object representing your desired kinematic tree.
2. Initialize the :code:`Topology`` with the desired initial configuration.
3. Define a `Sim` object using your :code:`Topology`, along with any desired 
    :code:`BodyDynamics` or :code:`JointDynamics` objects for applying forces to your 
    :code:`Topology`. Run the :code:`Sim`.
4. Analyze your sim data in one of two ways:
    1. Plot data from :code:`Sim.data_history`
    2. Visualize data with :code:`Visualizer`.

The following pages walk through an example.

.. toctree::
   :maxdepth: 1

   DefineATopology
   InitializeATopology
   DefineAndRunASim
   AnalyzeASim

