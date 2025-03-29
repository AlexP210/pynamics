============================
Define and Run a :code:`Sim`
============================

To define a :code:`Sim`, you need to provide a :code:`Topology`, and the dynamics
modules that will apply forces to the :code:`Topology`.

Dynamics Modules can be one of two types:

1. :code:`BodyDynamics`: Dynamics modules that compute cartesian force wrenches 
to act on bodies, expressed in each body's Identity frame.
2. :code:`JointDynamics`: Dynamics modules that compute joint-space forces to
act across each :code:`Joint`, expressed in the child frame of each :code:`Joint`.

Pynamics comes with some pre-made Dynamics Modules. For a list, see 
:doc:`../_autosummary/pynamics.dynamics`.

The code below combines our :code:`Topology` object from the previous page,
with three BodyDynamics modules, to initialize a :code:`Sim`. The first is a
:code:`Spring` module, pulling the "Cube"'s "Identity" frame to "World"'s "Identity"
frame, and the second is a :code:`QuadraticDrag` module to mimim the effect
of air resistance on the :code:`Cube` body as it bounces around, and the third
is a :code:`Gravity` module.

Note that the :code:`BodyDynamics` modules are provided as a dictionary, so
we can index the added Dynamics Modules by a unique string identifier.

.. code-block:: python

    from pynamics.sim import Sim
    from pynamics.dynamics import Gravity, Buoyancy, QuadraticDrag

    simulation = Sim(
        topology=topology,
        body_dynamics={
            "gravity": Gravity(
                g=-9.81, 
                direction=np.matrix([0,0,-1]).T, 
                body_names=["Cube",]
            ),
            "spring": Spring(
                body1="World", frame1="Identity",
                body2="Cube", frame2="Corner",
                stiffness=800
            ),
            "drag": QuadraticDrag(
                drag_models: trimesh.load(
                    file_obj=os.path.join("pynamics", "models", "common", "Cube.obj"),
                    file_type="obj", 
                    force="mesh",
                ),
                surface_point = np.matrix([0, 0, 1000]).T,
                fluid_density = 1.293,
            )
        }
    )

    simulation.simulate(delta_t=10, dt=0.01, verbose=True)

