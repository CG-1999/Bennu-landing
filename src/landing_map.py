import numpy as np
from scipy.spatial import cKDTree
import trimesh
from tqdm import tqdm
from vispy import scene
from vispy.scene import visuals
from vispy.color import get_colormap
from gravity_Intrep import GravityGrid

gravity_model = GravityGrid("bennu_gravity_field30.npz")

def normalize(x):
    x = np.asarray(x)
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-12)


def LandingMap_main(mesh_path="bennu.obj",
        g_field_path="bennu_gravity_field30.npz"):

    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    verts = mesh.vertices
    faces = mesh.faces

    face_centers = mesh.triangles_center
    face_normals = mesh.face_normals

    data = np.load(g_field_path)

    x = data["x"]
    y = data["y"]
    z = data["z"]

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    g_field = gravity_model.evaluate(points)

    batch_size = 5000
    g_surface_batches = []

    print("Evaluating gravity (Trilinear)...")

    for i in tqdm(range(0, len(face_centers), batch_size),
                desc="Evaluating gravity (Trilinear)"):

        batch = face_centers[i:i + batch_size]

        g_batch = gravity_model.evaluate(batch)

        g_surface_batches.append(g_batch)

    g_surface = np.vstack(g_surface_batches)
    g_mag = np.linalg.norm(g_surface, axis=1)

    # normal / tangential decomposition
    g_normal_mag = np.sum(g_surface * face_normals, axis=1)
    g_normal_vec = g_normal_mag[:, None] * face_normals

    g_tangential = g_surface - g_normal_vec
    g_tangential_mag = np.linalg.norm(g_tangential, axis=1)

    # Surface roughness
    k = 10
    tree1 = cKDTree(face_centers)
    _, idx2 = tree1.query(face_centers, k=k)

    roughness = np.zeros(len(face_centers))

    for i in tqdm(range(len(face_centers)), desc="Computing roughness"):
        n = face_normals[idx2[i]]
        roughness[i] = np.std(n)

    # Local gravity variation
    k_local = 20
    tree = cKDTree(points)
    _, idx3 = tree.query(face_centers, k=k_local)

    g_variation = np.zeros(len(face_centers))

    for i in tqdm(range(len(face_centers)), desc="Local gravity variation"):
        local_g = g_field[idx3[i]]
        g_variation[i] = np.mean(
            np.linalg.norm(local_g - np.mean(local_g, axis=0), axis=1)
        )

    # Slope
    gravity_dir = g_surface / (np.linalg.norm(g_surface, axis=1, keepdims=True) + 1e-12)

    dot = np.sum(face_normals * (-gravity_dir), axis=1)
    dot = np.clip(dot, -1.0, 1.0)

    slope = np.arccos(dot)

    # Normalization
    g_mag_n = normalize(g_mag)
    g_tan_n = normalize(g_tangential_mag)
    g_var_n = normalize(g_variation)
    roughness_n = normalize(roughness)
    slope_n = normalize(slope)

    # Cost function
    alpha = 1.0
    beta = 1.5
    gamma = 1.0
    delta = 0.5
    phi = 0.8

    landing_cost = (alpha * slope_n + beta * g_tan_n + gamma * g_var_n + delta * g_mag_n + phi * roughness_n)

    landing_idx = np.argmin(landing_cost)
    landing_point = face_centers[landing_idx]

    # Plot
    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
    view = canvas.central_widget.add_view()

    cmap = get_colormap('viridis')

    norm = (landing_cost - landing_cost.min()) / ((landing_cost.max() - landing_cost.min()) + 1e-12)
    colors = cmap.map(norm)

    mesh_vis = visuals.Mesh(vertices=verts, faces=faces, face_colors=colors)
    view.add(mesh_vis)

    marker = visuals.Markers()
    marker.set_data(
        np.array([landing_point]),
        face_color='red',
        size=10
    )
    view.add(marker)

    view.camera = scene.TurntableCamera(azimuth=30, elevation=30, distance=2000)

    canvas.app.run()
    np.savez("landing_point", landing_point=landing_point)
    return landing_point, landing_cost

if __name__ == "__main__":
    LandingMap_main()
