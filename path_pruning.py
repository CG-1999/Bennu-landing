import numpy as np

path = np.load("astar_path.npy")
data = np.load("bennu_occ.npz")
occupancy_grid = data['occupancy']

def line_free(p1, p2, n_samples=20):
    for t in np.linspace(0, 1, n_samples):
        p = (1 - t)*p1 + t*p2

        i, j, k = np.round(p).astype(int)

        if not (0 <= i < occupancy_grid.shape[0] and
               0 <= j < occupancy_grid.shape[1] and
               0 <= k < occupancy_grid.shape[2]):

               return False
        
        if occupancy_grid[i, j, k]:
             
             return False
        
    return True

def main_PathPruned():
    p_last = path[0]
    path_pruned = []
    path_pruned.append(p_last)
    candidate = path[1]
    for p in path[2:]:
        if line_free(p_last, p):
            candidate = p
        else:
            path_pruned.append(candidate)
            p_last = candidate
            candidate = p
            

    if not np.array_equal(path_pruned[-1], path[-1]):
        path_pruned.append(path[-1])

    path_pruned = np.array(path_pruned)
    np.save("bennu_path_pruned", path_pruned)
    return path_pruned

if __name__ == "__main__":
    main_PathPruned()






