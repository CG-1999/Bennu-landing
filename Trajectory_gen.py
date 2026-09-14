import numpy as np
from scipy.interpolate import CubicSpline
import sympy as sp

data = np.load("smooth_path.npz")
x = data["x"]
y = data["y"]
z = data["z"]

n_rows = x.shape[0]

def d_compute(p1, p2):
    return np.linalg.norm(p2 - p1)


def trj_main():
    s = np.zeros(n_rows)

    p_last = np.array([x[0], y[0], z[0]])

    for i in range(1, n_rows):
        p_curr = np.array([x[i], y[i], z[i]])
        s[i] = s[i - 1] + d_compute(p_last, p_curr)
        p_last = p_curr

    total_path_length = s[-1]

    # Normalize arc-length to [0, 1]
    if total_path_length > 0:
        s_norm = s / total_path_length
    else:
        raise ValueError("Total path length is zero. Invalid trajectory.")


    sx = CubicSpline(s_norm, x)
    sy = CubicSpline(s_norm, y)
    sz = CubicSpline(s_norm, z)

    dx_ds = sx.derivative(1)
    dy_ds = sy.derivative(1)
    dz_ds = sz.derivative(1)

    d2x_ds2 = sx.derivative(2)
    d2y_ds2 = sy.derivative(2)
    d2z_ds2 = sz.derivative(2)


    xi = sp.Symbol('xi')
    T = 100.0
    tau = xi / T

    # physical arc-length parameterization
    s_t = total_path_length * (10*tau**3 - 15*tau**4 + 6*tau**5)

    s_dot_t = sp.diff(s_t, xi)
    s_ddot_t = sp.diff(s_dot_t, xi)

    # convert to numeric functions
    s_func = sp.lambdify(xi, s_t, 'numpy')
    s_dot_func = sp.lambdify(xi, s_dot_t, 'numpy')
    s_ddot_func = sp.lambdify(xi, s_ddot_t, 'numpy')


    delta_t = 0.1
    t_vals = np.arange(0, T + delta_t, delta_t)

    s_tval = s_func(t_vals)
    s_dot_tval = s_dot_func(t_vals)
    s_ddot_tval = s_ddot_func(t_vals)

    s_tval = np.clip(s_tval, 0.0, total_path_length)

    # normalized lookup for splines
    s_lookup = s_tval / total_path_length

    x_t = sx(s_lookup)
    y_t = sy(s_lookup)
    z_t = sz(s_lookup)

    r_t = np.column_stack((x_t, y_t, z_t))

    vx = dx_ds(s_lookup) * (s_dot_tval / total_path_length)
    vy = dy_ds(s_lookup) * (s_dot_tval / total_path_length)
    vz = dz_ds(s_lookup) * (s_dot_tval / total_path_length)

    r_dot = np.column_stack((vx, vy, vz))

    ax = (d2x_ds2(s_lookup) * (s_dot_tval / total_path_length)**2 +
        dx_ds(s_lookup) * (s_ddot_tval / total_path_length))

    ay = (d2y_ds2(s_lookup) * (s_dot_tval / total_path_length)**2 +
        dy_ds(s_lookup) * (s_ddot_tval / total_path_length))

    az = (d2z_ds2(s_lookup) * (s_dot_tval / total_path_length)**2 +
        dz_ds(s_lookup) * (s_ddot_tval / total_path_length))

    r_ddot = np.column_stack((ax, ay, az))

    np.savez(
        "Nominal_pos_vel_acc_values.npz",
        r_t=r_t,
        r_dot=r_dot,
        r_ddot=r_ddot
    )

    print(f"Nominal states generated successfully. Trajectory distance: {total_path_length:.4f} meters.")

if __name__ == "__main__":
    trj_main()
