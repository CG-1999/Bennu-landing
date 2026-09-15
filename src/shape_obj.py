import numpy as np
from math import sqrt
import trimesh

#constants 
R0 = 245.28353613480672

# Legendre
def build_Pnm_dPnm(n_max, colat_arr):
    N = len(colat_arr)
    cos_th = np.cos(colat_arr)
    sin_th = np.sin(colat_arr)

    P, dP = {}, {}

    P[(0,0)] = np.ones(N)
    dP[(0,0)] = np.zeros(N)

    P[(1,0)] = sqrt(3)*cos_th
    dP[(1,0)] = -sqrt(3)*sin_th

    P[(1,1)] = sqrt(3)*sin_th
    dP[(1,1)] = sqrt(3)*cos_th

    for n in range(2, n_max+1):

        f = sqrt((2*n+1)/(2*n))
        P[(n,n)] = f*sin_th*P[(n-1,n-1)]
        dP[(n,n)] = f*(cos_th*P[(n-1,n-1)] + sin_th*dP[(n-1,n-1)])

        g = sqrt(2*n+1)
        P[(n,n-1)] = g*cos_th*P[(n-1,n-1)]
        dP[(n,n-1)] = g*(-sin_th*P[(n-1,n-1)] + cos_th*dP[(n-1,n-1)])

        for m in range(n-2, -1, -1):
            a = sqrt((4*n*n-1)/(n*n-m*m))
            b = sqrt((2*n+1)*(n-1-m)*(n-1+m)/((2*n-3)*(n*n-m*m)))

            P[(n,m)] = a*cos_th*P[(n-1,m)] - b*P[(n-2,m)]
            dP[(n,m)] = a*(-sin_th*P[(n-1,m)] + cos_th*dP[(n-1,m)]) - b*dP[(n-2,m)]

    return P


# load shape coefficients 
def load_coefficients(path):
    coeffs = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4:
                n, m = int(parts[0]), int(parts[1])
                coeffs[(n,m)] = (float(parts[2]), float(parts[3]))
    return coeffs


#  surface synthesis 
def synthesize_shape(coeffs, n_lat=360, n_lon=720):

    colat = np.linspace(0, np.pi, n_lat, endpoint=False) + np.pi/(2*n_lat)
    lons  = np.linspace(0, 2*np.pi, n_lon, endpoint=False)

    n_max = max(n for (n,_) in coeffs)
    P = build_Pnm_dPnm(n_max, colat)

    radius = np.zeros((n_lat, n_lon))

    for n in range(n_max+1):
        for m in range(n+1):
            if (n,m) not in coeffs:
                continue

            cnm, snm = coeffs[(n,m)]
            pnm = P[(n,m)]

            radius += np.outer(
                pnm,
                cnm*np.cos(m*lons) + snm*np.sin(m*lons)
            )

    return colat, lons, radius


#  convert to mesh 
def to_mesh(colat, lons, radius):

    LAT, LON = np.meshgrid(colat, lons, indexing="ij")

    x = radius*np.sin(LAT)*np.cos(LON)
    y = radius*np.sin(LAT)*np.sin(LON)
    z = radius*np.cos(LAT)

    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    faces = []

    n_lat, n_lon = radius.shape

    for i in range(n_lat-1):
        for j in range(n_lon-1):

            a = i*n_lon + j
            b = a + 1
            c = a + n_lon
            d = c + 1

            faces.append([3, a, b, c])
            faces.append([3, b, d, c])

    faces = np.array(faces).flatten()

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh


#  MAIN 
if __name__ == "__main__":

    shape_file = "bennu-shape-to15.txt"

    print("Loading coefficients...")
    coeffs = load_coefficients(shape_file)

    print("Building shape...")
    colat, lons, radius = synthesize_shape(coeffs)

    print("Creating mesh...")
    mesh = to_mesh(colat, lons, radius)

    print("Saving OBJ...")
    mesh.export("bennu.obj")

    print("Done → bennu.obj saved")
