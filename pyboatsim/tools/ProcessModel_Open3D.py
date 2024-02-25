# import open3d as o3d
# import numpy as np

# def get_edges(mesh):
#     pass

# def find_volume_below(mesh, cutoff, axis=2):

#     bounding_box = mesh.get_axis_aligned_bounding_box()
#     max_bound = np.array(bounding_box.max_bound)
#     min_bound = np.array(bounding_box.min_bound)
#     scale = (max_bound - min_bound)
#     center = 0.5*(max_bound + min_bound)

#     cropping_box_max_bound = max_bound.copy()
#     cropping_box_max_bound[axis] = cutoff
#     cropping_box_min_bound = min_bound.copy()
#     cropping_box_scale = cropping_box_max_bound - cropping_box_min_bound
#     cropping_box_center = center.copy()
#     cropping_box_center[axis] = min_bound[axis] + 0.5*cropping_box_scale[axis]

#     print(max_bound, min_bound, center, scale)
#     print(cropping_box_max_bound, cropping_box_min_bound, cropping_box_center, cropping_box_scale)
#     cropping_box = o3d.geometry.TriangleMesh.create_box(
#         width=3*float(cropping_box_scale[0]),
#         height=3*float(cropping_box_scale[1]),
#         depth=3*float(cropping_box_scale[2])
#     )
#     print(cropping_box.get_volume())
#     cropping_box.translate(-cropping_box_scale/2 + cropping_box_center)
#     cropping_box.compute_vertex_normals()
#     o3d.io.write_triangle_mesh("/mnt/c/Users/Alex/cropped.stl", cropping_box)

#     t_cropping_box = o3d.t.geometry.TriangleMesh.from_legacy(cropping_box)
#     t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
#     t_result = t_cropping_box.boolean_intersection(t_mesh, tolerance=0)
#     result = t_result.to_legacy()
    
#     result.remove_duplicated_vertices()
#     result.compute_vertex_normals()
#     o3d.io.write_triangle_mesh("/mnt/c/Users/Alex/result.stl", result)

#     return result.get_volume()

# mesh:o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(
#     filename="123CuboidTriangulated.stl",
#     enable_post_processing=True
# )
# mesh.remove_duplicated_vertices()
# print(mesh.get_volume())
# print(find_volume_below(mesh, 0))
