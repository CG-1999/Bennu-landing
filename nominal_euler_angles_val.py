import numpy as np
import matplotlib.pyplot as plt


def euler_nominal_main(current_orientation, blend_frac=0.05):

    data_traj = np.load('Nominal_pos_vel_acc_values.npz')
    r_dot = data_traj['r_dot']

    N = r_dot.shape[0]
    delta_t = 0.1
    t = np.arange(N) * delta_t
    T = t[-1]

    theta_start = current_orientation[0]
    psi_start   = current_orientation[1]
    phi_start   = current_orientation[2]

    vel_mag = np.linalg.norm(r_dot, axis=1)
    vmax = vel_mag.max()

    eps = 1e-3 * vmax  
    valid = vel_mag > eps

    if not np.any(valid):
        raise ValueError("Velocity magnitude is below threshold everywhere - "
                          "cannot determine a direction of travel.")

    safe_mag = np.where(valid, vel_mag, 1.0)
    u = r_dot / safe_mag[:, None]   

    idx = np.arange(N)
    valid_idx = idx[valid]
    first_valid, last_valid = valid_idx[0], valid_idx[-1]

    if first_valid > 0:
        u[:first_valid] = u[first_valid]
    if last_valid < N - 1:
        u[last_valid + 1:] = u[last_valid]

    if np.any(~valid[first_valid:last_valid + 1]):
        for axis in range(3):
            u[first_valid:last_valid + 1, axis] = np.interp(
                idx[first_valid:last_valid + 1],
                valid_idx,
                u[valid, axis]
            )
        norms = np.linalg.norm(u[first_valid:last_valid + 1], axis=1)
        u[first_valid:last_valid + 1] /= norms[:, None]

    phi_force = np.unwrap(np.arctan2(u[:, 1], u[:, 0]))
    psi_force = -np.arcsin(np.clip(u[:, 2], -1.0, 1.0))

    shift = 2 * np.pi * np.round((phi_force[0] - phi_start) / (2 * np.pi))
    phi_force -= shift

    win = max(3, int(0.01 * N) | 1)  # odd window, ~1% of N
    if win > 1:
        kernel = np.ones(win) / win
        pad = win // 2
        for arr in (phi_force, psi_force):
            padded = np.pad(arr, pad, mode='edge')
            arr[:] = np.convolve(padded, kernel, mode='valid')

    w = np.exp(-(t / (blend_frac * T)) ** 2)  # 1 at t=0, decays toward 0

    psi_val = (1 - w) * psi_force + w * psi_start
    phi_val = (1 - w) * phi_force + w * phi_start
    theta_val = np.full(N, theta_start, dtype=float)

    E = np.column_stack([theta_val, psi_val, phi_val])
    E_dot = np.gradient(E, delta_t, axis=0, edge_order=2)
    E_ddot = np.gradient(E_dot, delta_t, axis=0, edge_order=2)

    np.savez('Nominal_euler_angle_values.npz', E=E, E_dot=E_dot, E_ddot=E_ddot)

    print("Nominal attitude trajectory generated successfully.")
    print(f"E(0)   = {E[0]}")
    print(f"E(end) = {E[-1]}")
    print(f"max |E_dot|  = {np.max(np.abs(E_dot), axis=0)}")
    print(f"max |E_ddot| = {np.max(np.abs(E_ddot), axis=0)}")

    labels = ["Theta (Roll)", "Psi (Pitch)", "Phi (Yaw)"]

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    for i in range(3):
        plt.plot(t, E[:, i], label=labels[i], lw=2)
    plt.title("Euler Angles")
    plt.xlabel("Time [s]"); plt.ylabel("Angle [rad]")
    plt.grid(True); plt.legend()

    plt.subplot(2, 2, 2)
    for i in range(3):
        plt.plot(t, E_dot[:, i], label=labels[i], lw=2)
    plt.title("Angular Rates")
    plt.xlabel("Time [s]"); plt.ylabel("Rate [rad/s]")
    plt.grid(True); plt.legend()

    plt.subplot(2, 2, 3)
    for i in range(3):
        plt.plot(t, E_ddot[:, i], label=labels[i], lw=2)
    plt.title("Angular Accelerations")
    plt.xlabel("Time [s]"); plt.ylabel("Accel [rad/s^2]")
    plt.grid(True); plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(t, vel_mag, label="|v|", lw=2, color='k')
    plt.plot(t, w, label="blend weight w (toward current_orientation)", lw=2, color='r')
    plt.title("Velocity magnitude & start-blend weight")
    plt.xlabel("Time [s]")
    plt.grid(True); plt.legend()

    plt.tight_layout()
    plt.show()

    return E, E_dot, E_ddot


if __name__ == "__main__":
    c = np.array([0, 0, 0])
    euler_nominal_main(c)