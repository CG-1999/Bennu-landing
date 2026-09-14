import numpy as np
import trimesh
from heapq import heappush, heappop
from scipy.spatial import cKDTree
from tqdm import tqdm
from gravity_Intrep import GravityGrid

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="trimesh")

moves = [(i, j, k)
         for i in [-1, 0, 1]
         for j in [-1, 0, 1]
         for k in [-1, 0, 1]
         if not (i == 0 and j == 0 and k == 0)]


# Helpers
def load_occupancy(path):
    data = np.load(path)
    return data["occupancy"], data["x"], data["y"], data["z"]


def is_collision(occupancy, i, j, k):
    return occupancy[i, j, k]


def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def idx_to_pos(x, y, z, idx):
    i, j, k = idx
    return np.array([x[i], y[j], z[k]])


def nearest_index(x, y, z, p):
    return (np.argmin(np.abs(x - p[0])),
            np.argmin(np.abs(y - p[1])),
            np.argmin(np.abs(z - p[2])))


def astar(start, goal, cost_map, pos_grid, occupancy, x, y, z, step, landing_point):
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    pbar = tqdm(desc="A* search", unit=" node")

    while open_set:
        _, current = heappop(open_set)
        pbar.update(1)

        if current == goal:
            break

        ci, cj, ck = current
        p_curr = pos_grid[ci, cj, ck]

        for di, dj, dk in moves:
            ni, nj, nk = ci + di, cj + dj, ck + dk

            if not (0 <= ni < len(x) and 0 <= nj < len(y) and 0 <= nk < len(z)):
                continue

            if is_collision(occupancy, ni, nj, nk):
                continue

            neighbor = (ni, nj, nk)
            p_next = pos_grid[ni, nj, nk]

            move_length = step * np.linalg.norm([di, dj, dk])

            move_dir = p_next - p_curr
            move_dir /= (np.linalg.norm(move_dir) + 1e-12)

            to_land = landing_point - p_curr
            to_land /= (np.linalg.norm(to_land) + 1e-12)

            alignment = np.dot(move_dir, to_land)
            d_land = np.linalg.norm(p_curr - landing_point)

            align_penalty = (1 - alignment)
            align_weight = np.exp(-d_land / 800.0)

            direction_cost = 5.0 * align_weight * align_penalty * move_length

            step_cost = (
                move_length * 0.5 *
                (cost_map[ci, cj, ck] + cost_map[ni, nj, nk])
                + direction_cost
            )

            tentative_g = g_score[current] + step_cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f = tentative_g + heuristic(p_next, idx_to_pos(x, y, z, goal))
                heappush(open_set, (f, neighbor))

    pbar.close()

    if goal not in came_from:
        return [start]

    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])

    return path[::-1]


def plot_trajectory(mesh, path_xyz, start_point, hover_point, landing_point,
                     save_path="astar_trajectory.png"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the body (mesh) as a translucent surface
    verts = mesh.vertices[mesh.faces]
    mesh_collection = Poly3DCollection(verts, alpha=0.25, facecolor="gray",
                                        edgecolor="none")
    ax.add_collection3d(mesh_collection)

    # Plot the A* path
    ax.plot(path_xyz[:, 0], path_xyz[:, 1], path_xyz[:, 2],
            color="red", linewidth=2, label="A* path")

    # Markers
    ax.scatter(*start_point, color="blue", s=60, label="Start")
    ax.scatter(*hover_point, color="green", s=60, label="Goal (hover point)")
    ax.scatter(*landing_point, color="orange", s=60, label="Landing point")

    # Equal aspect ratio using mesh bounds
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    extent = (bounds[1] - bounds[0]).max() / 2
    ax.set_xlim(center[0] - extent, center[0] + extent)
    ax.set_ylim(center[1] - extent, center[1] + extent)
    ax.set_zlim(center[2] - extent, center[2] + extent)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("A* Trajectory")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()


# MAIN 
def astar_main(mesh_path="bennu.obj",
               g_field_path="bennu_gravity_field30.npz",
               occ_path="bennu_occ.npz",
               start_point=any,
               step=30):

    gravity_model = GravityGrid(g_field_path)

    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    occupancy, x, y, z = load_occupancy(occ_path)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    pos_grid = np.stack([X, Y, Z], axis=-1)

    landing_point = np.load("landing_point.npz")['landing_point']

    g_field = gravity_model.evaluate(np.column_stack([
        X.ravel(), Y.ravel(), Z.ravel()
    ]))

    g_grid = g_field.reshape(len(x), len(y), len(z), 3)
    g_mag = np.linalg.norm(g_grid, axis=3)
    g_mag /= (np.max(g_mag) + 1e-12)

    cost_map = g_mag.copy()
    cost_map[occupancy] = 1e9

    start_idx = nearest_index(x, y, z, start_point)

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

    hover_point = landing_point + 300.0 * landing_normal
    goal_idx = nearest_index(x, y, z, hover_point)

    path_idx = astar(
        start_idx,
        goal_idx,
        cost_map,
        pos_grid,
        occupancy,
        x, y, z,
        step,
        landing_point
    )

    path_xyz = np.array([idx_to_pos(x, y, z, i) for i in path_idx])

    np.save("astar_path.npy", path_xyz)

    plot_trajectory(mesh, path_xyz, start_point, hover_point, landing_point)

    return None


if __name__ == "__main__":
    astar_main()