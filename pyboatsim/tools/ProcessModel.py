import trimesh
import numpy as np

def get_submerged_volume(mesh, water_level, axis=2):
    min_bound, max_bound = mesh.bounds
    min_bound = np.array(min_bound)
    max_bound = np.array(max_bound)
    center = (min_bound + max_bound) / 2
    scale = max_bound - min_bound

    cropping_box_max_bound = max_bound.copy()
    cropping_box_max_bound[axis] = water_level
    cropping_box_min_bound = min_bound.copy()
    cropping_box_scale = cropping_box_max_bound - cropping_box_min_bound
    cropping_box_center = center.copy()
    cropping_box_center[axis] = min_bound[axis] + 0.5*cropping_box_scale[axis]

    transform = np.eye(4, 4)
    transform[0:3, 3] = cropping_box_center
    cropping_box = trimesh.primitives.Box(extents=cropping_box_scale, transform=transform)
    intersection = trimesh.boolean.intersection([mesh, cropping_box])

    cropping_box.export("/mnt/c/Users/Alex/CroppingBox.glb", "glb")
    intersection.export("/mnt/c/Users/Alex/Intersection.glb", "glb")

    return intersection

mesh:trimesh.Trimesh = trimesh.load(
    "/home/alex/Projects/PyBoAtSim/models/cup/cup_extruded.obj", 
    file_type="obj", 
    force="mesh",
    merge_norm=True,
    merge_tex=True )
# mesh.merge_vertices(digits_vertex=3)
print(mesh.vertices)
print(mesh.is_watertight)
mesh.show()