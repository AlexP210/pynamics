=====================
Analyze a :code:`Sim`
=====================

To analyze the results of a sim, we can do one of two things:

1. Inspect the contents of `Sim.data_history`
2. Use the :code:`Visualizer` to visualize our sim.

------------------------
Using :code:`Sim.data_history`
------------------------

After the :code:`Sim` runs, the `Sim.data_history` attribute can be accessed
to read data collected during the simulation. This attribute is a dictionary,
where keys are the data labels and the values are a time series.

Currently, Pynamics exposes the following data labels:

1. **"Time"**: The time in the simulation.
2. **"<Body Name> / Position <0-N>"**: For each Body Name, the joint-space 
    configuration of the body's parent :code:`Joint`.
3. **"<Body Name> / Velocity <0-N>"**: For each Body Name, the joint-space 
    velocity of the body's parent :code:`Joint`.

Going back to the example from last page, the code below plots the "Cube" 
body's distance from the origin over time.

.. code-block:: python

    import matplotlib.pyplot as plt
    import pandas as pd

    data = pd.DataFrame(simulation.data_history)
    distance_from_origin = (
        data["Cube / Position 4"] **2
        + data["Cube / Position 5"] **2
        + data["Cube / Position 6"] **2
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