import numpy as np
from scipy import interpolate
import trimesh
from scipy.spatial import cKDTree

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="trimesh")

def idx_to_pos(idx, x, y, z):
    i, j, k = idx
    return np.array([x[i], y[j], z[k]])

def load_grid_from_occ(path="bennu_occ.npz"):
    data = np.load(path)
    return data["x"], data["y"], data["z"]

def path_gen(path, k):
    path = np.array(path)
    tck, u = interpolate.splprep([path[:, 0], path[:, 1], path[:, 2]],s=0, k=k)
    u_fine = np.linspace(0, 1, 1000)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
    np.savez("smooth_path.npz", x=x_fine, y=y_fine, z=z_fine)
    return None 



def PathGeneration_main(mesh_path = "bennu.obj", start_point = any):
    mesh = trimesh.load(mesh_path)

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    landing_point = np.load(landing_point)['landing_point']
    face_centroids = mesh.triangles_center
    tree = cKDTree(face_centroids)

    radius = 0.05
    idx = tree.query_ball_point(landing_point, r=radius)

    normals = mesh.face_normals[idx]
    areas = mesh.area_faces[idx]

    weights = areas / (areas.sum() + 1e-12)

    surface_normal = np.average(normals, axis=0, weights=weights)
    surface_normal /= np.linalg.norm(surface_normal)

    center_to_point = landing_point - mesh.centroid
    center_to_point /= np.linalg.norm(center_to_point) + 1e-12

    if np.dot(surface_normal, center_to_point) < 0:
        surface_normal = -surface_normal

    phi = np.deg2rad(90)      # yaw around surface normal
    theta = np.deg2rad(-70)    # tilt from surface normal

    # local tangent basis
    t1 = np.cross(surface_normal, [0, 0, 1])

    if np.linalg.norm(t1) < 1e-6:
        t1 = np.cross(surface_normal, [1, 0, 0])

    t1 /= np.linalg.norm(t1)
    t2 = np.cross(surface_normal, t1)
    t2 /= np.linalg.norm(t2)

    landing_normal = (np.cos(theta) * surface_normal + np.sin(theta) * (np.cos(phi) * t1 + np.sin(phi) * t2))
    landing_normal /= np.linalg.norm(landing_normal)

    hover_height = 300.0 
    goal_point = landing_point + hover_height * landing_normal

    path_idx = np.load("bennu_path_pruned.npy")
    path_xyz = np.array([idx_to_pos(i) for i in path_idx])
    path_xyz[0] = start_point
    path_xyz[-1] = goal_point
    m = len(path_xyz)
    k = min(3, m - 1)
    path_gen(path_xyz, k)

if __name__ == "__main__":
    PathGeneration_main()