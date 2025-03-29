=========================
Define a :code:`Topology`
=========================

A :code:`Topology` is defined as a tree where the nodes are :code:`Body` objects,
representing rigid bodies in your system, and the edges are :code:`Joint` objects,
representing the articulated joints of the system.

-----------------------
Defining a :code:`Body`
-----------------------

A :code:`Body`, in turn, is defined by two things:

1. Its mass properties (total mass, center of mass, inertia tensor), which determine 
how it responds to forces.
2. Any co-ordinate frames of interest that exist on the :code:`Body`, which can
be used to attach other :code:`Body` objects through a :code:`Joint`, or to
apply a force.

To define both the mass properties of the :code:`Body`, and the co-ordinate frames that
reside on it we need a co-ordinate frame to begin with. Every body is defined with
a co-ordinate frame we can use to do this, called the body's "Identity" frame. The mass
properties, and any attached co-ordinate frames, are all defined with respect to the 
Identity frame.

Say we want to construct an 800kg, 2m x 2m x 2m cube, where we use the cube's center
as the identity frame to define it. There are to ways to do this in pynamics:

1. Explict assignment of mass properties:

    .. code-block:: python

        import numpy as np
        import pynamics.kinematics.topology as topo

        # 2x2x2 Cube of mass 800, defined based on the Identity frame being
        # at the center
        body = topo.Body(
            mass=800,
            center_of_mass=np.matrix([0, 0, 0]).T,
            # Cube inertia tensor
            inertia_matrix=(1/12)*800*(2**2+2**2)*np.eye(3,3)
        )

2. Using a mass properties model:
    * A "mass properties model" is simply a :code:`trimesh.Trimesh` object representing
        the geometry of a uniform density object. The origin of the model is assumed
        to be the Identity frame

    .. code-block:: python

        import os
        import trimesh
        import pynamics.kinematics.topology as topo

        body = topo.Body(
            mass_properties_model = trimesh.load(
                file_obj=os.path.join(
                    "pynamics", "models", "common", "Cube.obj"
                ),
                file_type="obj", 
                force="mesh"),
            density=100
        )

Option (1) is best suited for arbitrary objects with complicated density profiles.
Option (2) is best suited for uniform-density objects, where you have a 3D model
available.

To define co-ordinate frames on :code:`Body` objects, we use :code:`Frame` objects.
For example, to add a frame at the corner of the cube we just created above:

.. code-block:: python

    import numpy as np
    import pynamics.kinematics.topology as topo

    frame = topo.Frame(
        translation=np.matrix([1.0, 1.0, 1.0]).T, 
        rotation=np.eye(3,3)
    )

    body.add_frame(
        frame=frame,
        frame_name="Corner"
    )

Note that :code:`Frame`-s are assigned a :code:`frame_name` when added to a :code:`Body`.
This string will be used to index this frame on this body, going forward.

------------------------
Defining a :code:`Joint`
------------------------

Now that we have a :code:`Body`, we may want simulate it. That's the whole point,
after all to do so, we first need to incroporate it into a :code:`Topology`.

First, we need to create a :code:`Topology`:

.. code-block:: python

    import pynamics.kinematics.topology as topo

    topology = topo.Topology()

Every `Topology`` is created with a default immovable body, that you can start
connecting other :code:`Body` objects to. Similar to :code:`Frame`-s in a 
:code:`Body`, :code:`Body` objects in a :code:`Topology` are also indexed by name.
The default :code:`Body` in a :code:`Topology` is called "World".

To add our cube body from the previous section to the topology, using an
unconstrained joint:

.. code-block:: python
    
    import pynamics.kinematics.joint as joint
    
    topology.add_connection(
        parent_body_name="World",
        parent_frame_name="Identity,
        child_body=body,
        child_body_name="Cube",
        joint=joint.FreeJoint()
    )

Note that we added our body to the topology with a :code:`child_body_name` of "Cube".

Pynamics contains several other joint types, and more can be added. To see the
full list of joints pynamics supports out of the box, see :doc:`../_autosummary/pynamics.kinematics.joint`.
