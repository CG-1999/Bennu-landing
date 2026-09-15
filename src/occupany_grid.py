import numpy as np
import trimesh
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import rtree.index as _rtree_idx

if hasattr(_rtree_idx.Index, '__del__'):
    _orig_del = _rtree_idx.Index.__del__
    def _safe_del(self):
        try:
            _orig_del(self)
        except OSError:
            pass
    _rtree_idx.Index.__del__ = _safe_del

try:
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    USING_EMBREE = True
except Exception:
    from trimesh.ray.ray_triangle import RayMeshIntersector
    USING_EMBREE = False


def _cast_column(args):
 
    intersector, px, py, z_vals = args
    nz = len(z_vals)

    # Ray origin well below the mesh, direction straight up
    z_start = z_vals[0] - 1.0
    ray_origin = np.array([[px, py, z_start]], dtype=np.float64)
    ray_dir    = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)

    try:
        locs, _, _ = intersector.intersects_location(ray_origin, ray_dir,
                                                      multiple_hits=True)
    except Exception:
        return np.zeros(nz, dtype=bool)

    if len(locs) == 0:
        return np.zeros(nz, dtype=bool)

    # z-coordinates where the ray crosses the mesh surface, sorted ascending
    hit_z = np.sort(locs[:, 2])

    col = np.zeros(nz, dtype=bool)
    for k, zk in enumerate(z_vals):
        col[k] = (np.searchsorted(hit_z, zk, side='right') % 2) == 1

    return col


def build_and_save_occupancy_lowram(
    mesh_path,
    step=10,
    padding=1000,
    dilation_iter=3,
    save_path="occupancy_grid.npz",
    max_workers=None,         
    xy_batch_size=512):

    print("Loading mesh…")
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh.process(validate=True)

    intersector = RayMeshIntersector(mesh)
    backend = "Embree" if USING_EMBREE else "ray_triangle"
    print(f"Ray backend: {backend}")

    lo = mesh.bounds[0] - padding
    hi = mesh.bounds[1] + padding
    x = np.arange(lo[0], hi[0], step, dtype=np.float64)
    y = np.arange(lo[1], hi[1], step, dtype=np.float64)
    z = np.arange(lo[2], hi[2], step, dtype=np.float64)
    nx, ny, nz = len(x), len(y), len(z)

    print(f"Grid : {nx} × {ny} × {nz}  ({nx*ny*nz:,} voxels)")
    print(f"Columns to cast: {nx*ny:,}  |  workers: {max_workers or os.cpu_count()}")

    occupancy = np.zeros((nx, ny, nz), dtype=bool)

    # Build the full list of (ix, iy) pairs
    ix_all, iy_all = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    ix_flat = ix_all.ravel()
    iy_flat = iy_all.ravel()
    total_cols = len(ix_flat)

    workers = max_workers or os.cpu_count()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Submit in batches so tqdm updates smoothly without queuing all jobs
        with tqdm(total=total_cols, desc="Columns", unit="col") as pbar:
            for batch_start in range(0, total_cols, xy_batch_size):
                batch_end = min(batch_start + xy_batch_size, total_cols)

                futures = {
                    pool.submit(
                        _cast_column,
                        (intersector, x[ix_flat[k]], y[iy_flat[k]], z)
                    ): (ix_flat[k], iy_flat[k])
                    for k in range(batch_start, batch_end)
                }

                for fut in as_completed(futures):
                    ix_k, iy_k = futures[fut]
                    try:
                        occupancy[ix_k, iy_k, :] = fut.result()
                    except Exception as exc:
                        print(f"\nWarning: column ({ix_k},{iy_k}) failed: {exc}")
                    pbar.update(1)

    if dilation_iter > 0:
        print(f"Applying binary dilation ({dilation_iter} iterations)…")
        occupancy = binary_dilation(occupancy, iterations=dilation_iter)

    print("Saving…")
    np.savez_compressed(
        save_path,
        occupancy=occupancy,
        x=x, y=y, z=z,
        step=np.float64(step),
        padding=np.float64(padding),
    )
    filled = occupancy.sum()
    print(f"Saved: {save_path}  ({filled:,} / {occupancy.size:,} voxels filled, "
          f"{100*filled/occupancy.size:.1f} %)")
    return occupancy


if __name__ == "__main__":
    import sys
    mesh_path = sys.argv[1] if len(sys.argv) > 1 else "bennu.obj"
    build_and_save_occupancy_lowram(
        mesh_path=mesh_path, step=30, padding=1000, dilation_iter=3,
        save_path="occupancy_grid.npz", max_workers=None, xy_batch_size=256)
