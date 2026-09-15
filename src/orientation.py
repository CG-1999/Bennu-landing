import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

def nearest_index(x, y, z, p):
    return (np.argmin(np.abs(x - p[0])),
            np.argmin(np.abs(y - p[1])),
            np.argmin(np.abs(z - p[2])))

def o_main(mesh_path="bennu.obj"):

    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    landing_point = np.load("landing_point.npz")['landing_point']

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

    z_body = landing_normal
    x_body = t1
    y_body = np.cross(z_body, x_body)

    x_body /= np.linalg.norm(x_body)
    y_body /= np.linalg.norm(y_body)

    R = np.column_stack((x_body, y_body, z_body))

    rot = Rot.from_matrix(R)

    phi_deg, theta_deg, psi_deg = rot.as_euler('XYZ', degrees=True)

    return np.radians(theta_deg), np.radians(psi_deg), np.radians(phi_deg)

