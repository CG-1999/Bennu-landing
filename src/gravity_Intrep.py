import numpy as np

class GravityGrid:
    def __init__(self, path):
        data = np.load(path)

        self.gx = data["gx"]
        self.gy = data["gy"]
        self.gz = data["gz"]

        self.x = data["x"]
        self.y = data["y"]
        self.z = data["z"]

        self.nx = len(self.x)
        self.ny = len(self.y)
        self.nz = len(self.z)

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]

    def evaluate(self, points):
        """
        points: (N,3)
        returns: (N,3)
        """

        gx_out = np.zeros(len(points))
        gy_out = np.zeros(len(points))
        gz_out = np.zeros(len(points))

        for i, (x, y, z) in enumerate(points):

            ix = np.clip(int((x - self.x[0]) / self.dx), 0, self.nx - 2)
            iy = np.clip(int((y - self.y[0]) / self.dy), 0, self.ny - 2)
            iz = np.clip(int((z - self.z[0]) / self.dz), 0, self.nz - 2)

            tx = (x - self.x[ix]) / self.dx
            ty = (y - self.y[iy]) / self.dy
            tz = (z - self.z[iz]) / self.dz

            # 8 corners
            def g(ix, iy, iz, field):
                return field[ix, iy, iz]

            c000 = np.array([g(ix, iy, iz, self.gx),
                             g(ix, iy, iz, self.gy),
                             g(ix, iy, iz, self.gz)])

            c100 = np.array([g(ix+1, iy, iz, self.gx),
                             g(ix+1, iy, iz, self.gy),
                             g(ix+1, iy, iz, self.gz)])

            c010 = np.array([g(ix, iy+1, iz, self.gx),
                             g(ix, iy+1, iz, self.gy),
                             g(ix, iy+1, iz, self.gz)])

            c110 = np.array([g(ix+1, iy+1, iz, self.gx),
                             g(ix+1, iy+1, iz, self.gy),
                             g(ix+1, iy+1, iz, self.gz)])

            c001 = np.array([g(ix, iy, iz+1, self.gx),
                             g(ix, iy, iz+1, self.gy),
                             g(ix, iy, iz+1, self.gz)])

            c101 = np.array([g(ix+1, iy, iz+1, self.gx),
                             g(ix+1, iy, iz+1, self.gy),
                             g(ix+1, iy, iz+1, self.gz)])

            c011 = np.array([g(ix, iy+1, iz+1, self.gx),
                             g(ix, iy+1, iz+1, self.gy),
                             g(ix, iy+1, iz+1, self.gz)])

            c111 = np.array([g(ix+1, iy+1, iz+1, self.gx),
                             g(ix+1, iy+1, iz+1, self.gy),
                             g(ix+1, iy+1, iz+1, self.gz)])

            # trilinear interpolation
            c00 = c000*(1-tx) + c100*tx
            c01 = c001*(1-tx) + c101*tx
            c10 = c010*(1-tx) + c110*tx
            c11 = c011*(1-tx) + c111*tx

            c0 = c00*(1-ty) + c10*ty
            c1 = c01*(1-ty) + c11*ty

            c = c0*(1-tz) + c1*tz

            gx_out[i], gy_out[i], gz_out[i] = c

        return np.column_stack([gx_out, gy_out, gz_out])
