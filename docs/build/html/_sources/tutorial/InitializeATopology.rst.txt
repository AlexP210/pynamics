=============================
Initialize a :code:`Topology`
=============================

The :code:`Joint` objects in a :code:`Topology` can be articulated to define the
initial configuration you want to use for your simulation. 

You can access the :code:`Joint` objects in a :code:`Topology` through the
:code:`Topology.joints` attribute. Since :code:`Topology` can only be a kinematic
tree, each :code:`Joint` is uniquely indexed by its child body.

Setting the configuration of a :code:`Joint` in the way you want requires knowing
the representation used by the :code:`Joint`. For all the joints included in 
default pynamics, this is listed in the Notes section under each joints documentation
(see :doc:`../_autosummary/pynamics.kinematics.joint`).

To set an initial articulation and joint-space velocity of a :code:`Joint`, we use
:code:`Joint.set_configuration` and :code:`Joint.set_configuration_d` respectively.

For example, to articulate the :code:`FreeJoint` from oue example by +1m in X, 
and add an initial velocity of +1 m/s in Y:

.. code-block:: python
        
    topology.joints["Cube"].set_configuration(1, 0, 0, 0, 1, 0, 0)
    topology.joints["Cube"].set_configuration_d(0, 0, 0, 0, 0, 1, 0)

