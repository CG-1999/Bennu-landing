import numpy as np
import trimesh
from heapq import heappush, heappop
from scipy.spatial import cKDTree
from tqdm import tqdm
from gravity_Intrep import GravityGrid
import plotly.graph_objects as go

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
                     save_path="astar_trajectory.html"):
    verts = mesh.vertices
    faces = mesh.faces

    mesh_trace = go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color="lightgray", opacity=0.5, flatshading=True,
        name="Body", showscale=False,
    )

    path_trace = go.Scatter3d(
        x=path_xyz[:, 0], y=path_xyz[:, 1], z=path_xyz[:, 2],
        mode="lines", line=dict(color="red", width=5),
        name="A* path",
    )

    # Surface normal, drawn as a line from the landing point to the hover
    # point, with a cone arrowhead showing direction.
    normal_vec = np.array(hover_point) - np.array(landing_point)

    normal_line = go.Scatter3d(
        x=[landing_point[0], hover_point[0]],
        y=[landing_point[1], hover_point[1]],
        z=[landing_point[2], hover_point[2]],
        mode="lines", line=dict(color="purple", width=4, dash="dash"),
        name="Surface normal",
    )

    normal_arrowhead = go.Cone(
        x=[hover_point[0]], y=[hover_point[1]], z=[hover_point[2]],
        u=[normal_vec[0]], v=[normal_vec[1]], w=[normal_vec[2]],
        sizemode="absolute", sizeref=20, anchor="tip",
        colorscale=[[0, "purple"], [1, "purple"]], showscale=False,
        name="Surface normal",
    )

    def marker(point, color, label):
        return go.Scatter3d(
            x=[point[0]], y=[point[1]], z=[point[2]],
            mode="markers", marker=dict(color=color, size=6),
            name=label,
        )

    fig = go.Figure(data=[
        mesh_trace,
        path_trace,
        normal_line,
        normal_arrowhead,
        marker(start_point, "blue", "Start"),
        marker(hover_point, "green", "Goal (hover point)"),
        marker(landing_point, "orange", "Landing point"),
    ])

    fig.update_layout(
        title="A* Trajectory",
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig.write_html(save_path)
    fig.show()


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
    theta = np.deg2rad(-20)    # tilt from surface normal

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