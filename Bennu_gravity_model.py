import argparse, os, sys
import numpy as np
from math import sqrt
from tqdm import tqdm
from numba import njit, prange


G   = 6.67384e-11
M   = 7.8e10
R0  = 245.28353613480672
GM  = G * M


@njit(cache=True, parallel=True)
def _build_Pnm_kernel(n_max: int, cos_th: np.ndarray, sin_th: np.ndarray):

    N      = cos_th.shape[0]
    n_terms = (n_max + 1) * (n_max + 2) // 2   # total (n,m) pairs
    P  = np.zeros((n_terms, N))
    dP = np.zeros((n_terms, N))

    # Helper: flat index for (n, m)
    # idx(n,m) = n*(n+1)//2 + m

    # Seed values (independent of θ), initialise before prange
    P [0, :]  = 1.0          # (0,0)
    # dP[0,:] = 0  already

    for i in prange(N):
        c = cos_th[i]
        s = sin_th[i]

        # (1,0)
        sq3 = 1.7320508075688772  # sqrt(3)
        P [1, i]  =  sq3 * c
        dP[1, i]  = -sq3 * s
        # (1,1)
        P [2, i]  =  sq3 * s
        dP[2, i]  =  sq3 * c

        for n in range(2, n_max + 1):
            base = n * (n + 1) // 2

            # sectoral (n,n)
            f = sqrt((2.0*n + 1.0) / (2.0*n))
            prev_sec = (n-1)*n//2 + (n-1)  # idx(n-1,n-1)
            P [base + n, i]  = f * s * P [prev_sec, i]
            dP[base + n, i]  = f * (c * P[prev_sec, i] + s * dP[prev_sec, i])

            # near-sectoral (n, n-1)
            g = sqrt(2.0*n + 1.0)
            P [base + n-1, i]  = g * c * P [prev_sec, i]
            dP[base + n-1, i]  = g * (-s * P[prev_sec, i] + c * dP[prev_sec, i])

            # general (n, m) for m = n-2 .. 0
            for m in range(n-2, -1, -1):
                a  = sqrt((4.0*n*n - 1.0) / (n*n - m*m))
                b  = sqrt((2.0*n+1.0) * (n-1.0-m) * (n-1.0+m) /
                          ((2.0*n-3.0) * (n*n - m*m)))
                pm1 = (n-1)*n//2 + m          # idx(n-1,m)
                pm2 = (n-2)*(n-1)//2 + m      # idx(n-2,m)
                P [base + m, i]  = a*c*P [pm1,i] - b*P [pm2,i]
                dP[base + m, i]  = a*(-s*P[pm1,i] + c*dP[pm1,i]) - b*dP[pm2,i]

    return P, dP


def build_Pnm_arrays(n_max: int, colat: np.ndarray):

    cos_th = np.cos(colat).astype(np.float64)
    sin_th = np.sin(colat).astype(np.float64)
    P, dP = _build_Pnm_kernel(n_max, cos_th, sin_th)
    return P, dP


def _idx(n, m):
    return n * (n + 1) // 2 + m


def build_Pnm_dPnm(n_max: int, colat_arr: np.ndarray):
    P_arr, dP_arr = build_Pnm_arrays(n_max, colat_arr)
    P  = {(n, m): P_arr[_idx(n,m)]  for n in range(n_max+1) for m in range(n+1)}
    dP = {(n, m): dP_arr[_idx(n,m)] for n in range(n_max+1) for m in range(n+1)}
    return P, dP


# I/O
def load_coefficients(path: str) -> dict:
    coeffs = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 4:
                n, m = int(parts[0]), int(parts[1])
                coeffs[(n, m)] = (float(parts[2]), float(parts[3]))
    return coeffs

# Shape synthesis
def synthesise_radius(coeffs: dict, n_lat: int = 360, n_lon: int = 720):
    colat  = np.linspace(0, np.pi,    n_lat, endpoint=False) + np.pi/(2*n_lat)
    lons   = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    n_max  = max(n for (n,_) in coeffs)

    print(f"  Building P̄nm table (n_max={n_max}, n_lat={n_lat}) …")
    P_arr, _ = build_Pnm_arrays(n_max, colat)  # (n_terms, n_lat)

    # Pre-build trig arrays for all m at once
    m_vals = np.arange(n_max + 1)
    cos_ml = np.cos(np.outer(m_vals, lons))    # (n_max+1, n_lon)
    sin_ml = np.sin(np.outer(m_vals, lons))

    radius = np.zeros((n_lat, n_lon))
    for n in range(n_max + 1):
        for m in range(n + 1):
            if (n, m) not in coeffs:
                continue
            cnm, snm = coeffs[(n, m)]
            pnm = P_arr[_idx(n, m)]                          # (n_lat,)
            trig = cnm * cos_ml[m] + snm * sin_ml[m]        # (n_lon,)
            radius += np.outer(pnm, trig)

    return 90.0 - np.degrees(colat), np.degrees(lons), radius


def radius_to_xyz(lats, lons, radius):
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)
    LAT, LON = np.meshgrid(lat_r, lon_r, indexing="ij")
    x = radius * np.cos(LAT) * np.cos(LON)
    y = radius * np.cos(LAT) * np.sin(LON)
    z = radius * np.sin(LAT)
    return x, y, z


def build_surface_mesh(x, y, z, radius):
    import pyvista as pv
    grid = pv.StructuredGrid(x, y, z)
    grid["radius_m"] = radius.ravel()
    return grid


def synthesise_potential_and_gradient(pot_coeffs: dict,
                                      colat: np.ndarray,
                                      lons:  np.ndarray,
                                      r: float,
                                      n_max_use: int = 450):
    n_max  = min(n_max_use, max(n for (n,_) in pot_coeffs))
    n_lat, n_lon = len(colat), len(lons)

    print(f"  Building P̄nm+dP̄nm table (n_max={n_max}, n_lat={n_lat}) …")
    P_arr, dP_arr = build_Pnm_arrays(n_max, colat)   # (n_terms, n_lat)

    # Pre-compute trig
    m_vals = np.arange(n_max + 1)
    cos_ml = np.cos(np.outer(m_vals, lons))           # (n_max+1, n_lon)
    sin_ml = np.sin(np.outer(m_vals, lons))

    V   = np.zeros((n_lat, n_lon))
    gr  = np.zeros((n_lat, n_lon))
    gth = np.zeros((n_lat, n_lon))
    gla = np.zeros((n_lat, n_lon))

    # Group by degree n so we compute the radial factor once per n
    for n in range(n_max + 1):
        radial    = (R0 / r) ** (n + 1)
        radial_dr = -(n + 1) / r * radial

        for m in range(n + 1):
            if (n, m) not in pot_coeffs:
                continue
            cnm, snm = pot_coeffs[(n, m)]
            pnm  = P_arr [_idx(n, m)]                        # (n_lat,)
            dpnm = dP_arr[_idx(n, m)]                        # (n_lat,)
            trig  = cnm * cos_ml[m] + snm * sin_ml[m]       # (n_lon,)
            dtrig = m * (-cnm * sin_ml[m] + snm * cos_ml[m])

            V   += radial    * np.outer(pnm,  trig)
            gr  += radial_dr * np.outer(pnm,  trig)
            gth += radial    * np.outer(dpnm, trig)
            gla += radial    * np.outer(pnm,  dtrig)

    prefac = GM / R0
    V   *= prefac
    gr  *= prefac
    gth *= prefac / r
    gla *= prefac / r

    sin_c = np.maximum(np.sin(colat), 1e-10)
    gla   = (gla.T / sin_c).T

    return V, gr, gth, gla


def gradient_to_cartesian(colat, lons, gr, gth, gla):
    TH, LA = np.meshgrid(colat, lons, indexing="ij")
    st = np.sin(TH); ct = np.cos(TH)
    sl = np.sin(LA); cl = np.cos(LA)
    gx = gr*st*cl + gth*ct*cl - gla*sl
    gy = gr*st*sl + gth*ct*sl + gla*cl
    gz = gr*ct    - gth*st
    return gx, gy, gz


def gravity_at_xyz(x, y, z, pot_coeffs, n_max, P_cache=None):
    
    scalar = np.ndim(x) == 0
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    z = np.atleast_1d(np.asarray(z, dtype=np.float64))

    r     = np.sqrt(x*x + y*y + z*z)
    # Avoid divide-by-zero
    safe_r = np.where(r < 1e-6, 1.0, r)

    theta = np.arccos(np.clip(z / safe_r, -1.0, 1.0))   # (N,)
    phi   = np.mod(np.arctan2(y, x), 2*np.pi)            # (N,)

    # Build P table over all distinct colatitudes at once
    P_arr, dP_arr = build_Pnm_arrays(n_max, theta)       # (n_terms, N)

    # Collect coefficient arrays in degree order
    nm_pairs = [(n, m)
                for n in range(n_max + 1)
                for m in range(n + 1)
                if (n, m) in pot_coeffs]

    ns   = np.array([n for n, m in nm_pairs], dtype=np.int32)
    ms   = np.array([m for n, m in nm_pairs], dtype=np.int32)
    cnms = np.array([pot_coeffs[(n,m)][0] for n,m in nm_pairs])
    snms = np.array([pot_coeffs[(n,m)][1] for n,m in nm_pairs])
    idxs = np.array([_idx(n, m) for n, m in nm_pairs], dtype=np.int32)

    # Radial factors  shape (n_terms, N)
    radial    = (R0 / safe_r[np.newaxis, :]) ** (ns[:, np.newaxis] + 1)
    radial_dr = -(ns[:, np.newaxis] + 1) / safe_r[np.newaxis, :] * radial

    # Legendre values  shape (n_terms, N)
    pnm  = P_arr [idxs, :]    # (n_terms, N)
    dpnm = dP_arr[idxs, :]

    unique_ms = np.arange(n_max + 1)
    cos_by_m  = np.cos(np.outer(unique_ms, phi))   # (n_max+1, N)
    sin_by_m  = np.sin(np.outer(unique_ms, phi))
    cos_ml = cos_by_m[ms]     # (n_terms, N)
    sin_ml = sin_by_m[ms]
    trig   = cnms[:, np.newaxis] * cos_ml + snms[:, np.newaxis] * sin_ml
    dtrig  = ms[:, np.newaxis] * (-cnms[:, np.newaxis] * sin_ml
                                  + snms[:, np.newaxis] * cos_ml)

    # Sum over (n_terms) axis → (N,)
    gr  = np.sum(radial_dr * pnm  * trig,  axis=0)
    gth = np.sum(radial    * dpnm * trig,  axis=0)
    gla = np.sum(radial    * pnm  * dtrig, axis=0)

    prefac = GM / R0
    gr  *= prefac
    gth *= prefac / safe_r
    gla *= prefac / safe_r

    sin_theta = np.maximum(np.sin(theta), 1e-10)
    gla /= sin_theta

    # Spherical → Cartesian
    st = np.sin(theta); ct = np.cos(theta)
    sp = np.sin(phi);   cp = np.cos(phi)

    gx_out = gr*st*cp + gth*ct*cp - gla*sp
    gy_out = gr*st*sp + gth*ct*sp + gla*cp
    gz_out = gr*ct    - gth*st

    # Zero out points too close to origin
    mask = r < 1e-6
    gx_out[mask] = 0.0
    gy_out[mask] = 0.0
    gz_out[mask] = 0.0

    if scalar:
        return float(gx_out[0]), float(gy_out[0]), float(gz_out[0])
    return gx_out, gy_out, gz_out


# CLI
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shape",     default="bennu-shape-to15.txt")
    p.add_argument("--potential", default="bennu-potential-to450-ie.txt")
    p.add_argument("--altitude",  type=float, default=500.0)
    p.add_argument("--grid",      type=int, nargs=2, default=[72, 144],
                   metavar=("N_LAT","N_LON"))
    p.add_argument("--subsample", type=int, default=1)
    p.add_argument("--nmax",      type=int, default=450)
    args, _ = p.parse_known_args()
    return args


def find_file(name):
    for c in [name, os.path.join(os.path.dirname(os.path.abspath(__file__)), name)]:
        if os.path.exists(c):
            return c
    print(f"ERROR: Cannot find '{name}'.", file=sys.stderr)
    sys.exit(1)


def Bennu_main(field_range):
    args = parse_args()

    # shape surface
    print("Loading shape coefficients …")
    shape_coeffs = load_coefficients("bennu-shape-to15.txt")
    n_shape = max(n for (n,_) in shape_coeffs)
    print(f"  n_max={n_shape}, {len(shape_coeffs)} terms")

    print("Synthesising surface (360 × 720 grid) …")
    lats, lons_deg, radius = synthesise_radius(shape_coeffs, 360, 720)
    print(f"  radius range: {radius.min():.1f} – {radius.max():.1f} m")

    x, y, z = radius_to_xyz(lats, lons_deg, radius)
    surface_mesh = build_surface_mesh(x, y, z, radius)

    # gravity probe shell
    print("Loading potential coefficients …")
    pot_coeffs = load_coefficients("bennu-potential-to450-ie.txt")

    bounds = surface_mesh.bounds
    pad  = np.round(field_range / R0)
    step = 30

    x1 = np.arange(bounds[0] - pad*R0, bounds[1] + pad*R0 + step, step)
    y1 = np.arange(bounds[2] - pad*R0, bounds[3] + pad*R0 + step, step)
    z1 = np.arange(bounds[4] - pad*R0, bounds[5] + pad*R0 + step, step)
    nx, ny, nz = len(x1), len(y1), len(z1)

    print("Computing global field (n=30) …")
    xx, yy, zz = np.meshgrid(x1, y1, z1, indexing='ij')
    pts  = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    N_g  = len(pts)


    CHUNK_G = 5_000
    gx_list, gy_list, gz_list = [], [], []
    for start in tqdm(range(0, N_g, CHUNK_G), desc="Global field"):
        sl = slice(start, start + CHUNK_G)
        a, b, c = gravity_at_xyz(pts[sl,0], pts[sl,1], pts[sl,2], pot_coeffs, 30)
        gx_list.append(a); gy_list.append(b); gz_list.append(c)
    gx_grid = np.concatenate(gx_list).reshape(nx, ny, nz)
    gy_grid = np.concatenate(gy_list).reshape(nx, ny, nz)
    gz_grid = np.concatenate(gz_list).reshape(nx, ny, nz)

    np.savez_compressed("bennu_gravity_field30.npz",
                        x=x1, y=y1, z=z1,
                        gx=gx_grid, gy=gy_grid, gz=gz_grid)

    # near-surface field
    near_step = 30
    rmax      = R0 + 500
    x2 = np.arange(-rmax, rmax + near_step, near_step)
    y2 = np.arange(-rmax, rmax + near_step, near_step)
    z2 = np.arange(-rmax, rmax + near_step, near_step)

    xx2, yy2, zz2 = np.meshgrid(x2, y2, z2, indexing='ij')
    xx2, yy2, zz2 = xx2.ravel(), yy2.ravel(), zz2.ravel()

    r2       = np.sqrt(xx2**2 + yy2**2 + zz2**2)
    altitude = r2 - R0
    mask     = (altitude >= 0) & (altitude <= 500)
    xx2, yy2, zz2 = xx2[mask], yy2[mask], zz2[mask]

    print(f"Computing near-surface field (n=450), {len(xx2):,} points …")
    CHUNK = 500
    gx_list, gy_list, gz_list = [], [], []
    for start in tqdm(range(0, len(xx2), CHUNK), desc="Near-surface"):
        sl = slice(start, start + CHUNK)
        gx_c, gy_c, gz_c = gravity_at_xyz(xx2[sl], yy2[sl], zz2[sl],
                                           pot_coeffs, 450)
        gx_list.append(gx_c); gy_list.append(gy_c); gz_list.append(gz_c)

    np.savez_compressed("bennu_gravity_near_n450.npz",
                        x=xx2, y=yy2, z=zz2,
                        gx=np.concatenate(gx_list),
                        gy=np.concatenate(gy_list),
                        gz=np.concatenate(gz_list))