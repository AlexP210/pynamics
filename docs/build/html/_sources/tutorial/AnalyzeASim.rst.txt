=====================
Analyze a :code:`Sim`
=====================

To analyze the results of a sim, we can do one of two things:

1. Inspect the contents of `Sim.data_history`
2. Use the :code:`Visualizer` to visualize our sim.

----------------------
Using :code:`Sim.data`
----------------------

After the :code:`Sim` runs, the `Sim.data` attribute can be accessed
to read data collected during the simulation. This attribute is a nested 
dictionary, with the following structure:

- "Time"
- "Bodies"
    - <Body Name>
        - "Position [0-3]": ("World", "Identity") to ("Body","Identity") quaternion,
            with (w, x, y, z) format
        - "Position [4-6]": ("World", "Identity") to ("Body","Identity") translation
            vector, expressed in ("World", "Identity")
        - "Velocity [0-5]": Body velocity in ("World", "Identity") frame, in [w, z] format.
        - "Acceleration [0-5]": Body acceleration in ("World", "Identity") frame, in [\alpha, a] format.
- "Joints"
    - <Joint Name>
        - "Position [0-N]": Position co-ordinates in joint space.
        - "Velocity [0-N]": Velocity in joint space.
        - "Acceleration [0-N]": Acceleration in joint space.
- "Body Forces"
    - <Force Module Name>
        - <Body Name>
            - "Total Force": Magnitude of the applied force vector.
            - "Total Moment": Magnitude of the applied force moment/torque.
            - "Force [0-5]": Components of the torque-force wrench expressed in
                the ("World", "Identity") frame.
            - <Additional Data Labels>: Additional data determined by the force module.
    
Going back to the example from last page, the code below plots the "Cube" 
body's distance from the origin over time.

.. code-block:: python

    import matplotlib.pyplot as plt
    import pandas as pd

    distance_from_origin = (
        sim.data["Bodies"]["Cube"]["Position 4"] **2
        sim.data["Bodies"]["Cube"]["Position 5"] **2
        sim.data["Bodies"]["Cube"]["Position 6"] **2
    )**(0.5)

    plt.plot(data["Time"], distance)
    plt.xlabel("Time (s)")
    plt.ylabel("Distance From Origin to Cube (m)")
    plt.show()
    
------------------------
Using :code:`Visualizer`
------------------------

Alternatively, if you want a graphical depiciton of your sim, you can use the
:code:`Visualizer` class. To use this, you first need a "visualization model"
for each body in your simulation. This is a :code:`trimesh.Trimesh` object that
represents how your object looks. 

To set up a :code:`Visualizer` instance for our sim from before:

.. code-block:: python

    from pynamics.visualizer import Visualizer

    visualizer = Visualizer(
        topology=topology,
        visualization_models={
            ("Cube", "Identity"): trimesh.load(
                file_obj=os.path.join("pynamics", "models", "common", "Cube.obj"),
                file_type="obj", 
                force="mesh",
            )
        },
        sim=simulation
    )

    visualizer.animate(save_path="example_video.mp4", verbose=True)

In this case, the origin of the visualization model lines up with the "Identity"
frame we've decided to use for the "Cube" body. If this was not the case,
replace "Identity" with whatever frame name on the :code:`Body` you want to attach
the visualization model to.