import numpy as np
import matplotlib.pyplot as plt

# dynamics
def dynamics(x, k, A_all_ext, B_all_ext, K_all_ext, ref_ext, u_all_ext):
    A = A_all_ext[k]
    B = B_all_ext[k]
    K = K_all_ext[k]

    x_ref = ref_ext[k]
    u_ref = u_all_ext[k]

    u = u_ref - K @ (x - x_ref)

    return A @ x + B @ u

def controller_main(dt, extend_time_sec):

    # load data
    data = np.load("linearised_matrices.npz")
    A_all = data["A"]
    B_all = data["B"]

    data_traj = np.load("Nominal_pos_vel_acc_values.npz")
    r_t = data_traj["r_t"]
    r_dot = data_traj["r_dot"]

    data_traj_E = np.load("Nominal_euler_angle_values.npz")
    E = data_traj_E["E"]
    E_dot = data_traj_E["E_dot"]

    K_all = np.load("K_matrices.npz")["K"]
    u_all = np.load("Nominal_control_input.npz")["u_r"]

    # time
    extend_steps = int(extend_time_sec / dt)

    N_base = r_t.shape[0]
    N = N_base + extend_steps

    # reference trajectory
    ref = np.zeros((N_base, 12))
    ref[:, 0:3] = r_t
    ref[:, 3:6] = E
    ref[:, 6:9] = r_dot
    ref[:, 9:12] = E_dot

    # hover extension
    ref_last = ref[-1]
    ref_ext = np.vstack([ref,np.tile(ref_last, (extend_steps, 1))])


    # extend system matrix
    A_last = A_all[-1]
    B_last = B_all[-1]
    K_last = K_all[-1]
    u_last = u_all[-1]

    A_all_ext = np.vstack([
        A_all,
        np.repeat(A_last[None, :, :], extend_steps, axis=0)
    ])

    B_all_ext = np.vstack([
        B_all,
        np.repeat(B_last[None, :, :], extend_steps, axis=0)
    ])

    K_all_ext = np.vstack([
        K_all,
        np.repeat(K_last[None, :, :], extend_steps, axis=0)
    ])

    u_all_ext = np.vstack([
        u_all,
        np.repeat(u_last[None, :], extend_steps, axis=0)
    ])


    # rk4
    y = np.zeros((N, 12))
    y[0] = ref_ext[0]

    for k in range(N - 1):

        x = y[k]

        k1 = dynamics(x, k, A_all_ext, B_all_ext, K_all_ext, ref_ext, u_all_ext)
        k2 = dynamics(x + 0.5 * dt * k1, k, A_all_ext, B_all_ext, K_all_ext, ref_ext, u_all_ext)
        k3 = dynamics(x + 0.5 * dt * k2, k, A_all_ext, B_all_ext, K_all_ext, ref_ext, u_all_ext)
        k4 = dynamics(x + dt * k3, k, A_all_ext, B_all_ext, K_all_ext, ref_ext, u_all_ext)

        y[k + 1] = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        if np.any(np.isnan(y[k + 1])) or np.any(np.isinf(y[k + 1])):
            print(f"Divergence detected at step {k}")
            y = y[:k+1]
            break

    print("done:", y.shape)


    # error computation
    ref_extended = ref_ext[:len(y)]
    error = y - ref_extended
    error_norm_mat = np.linalg.norm(error, axis=1)

    # plotting
    state_labels = [
        "x", "y", "z",
        "roll", "pitch", "yaw",
        "vx", "vy", "vz",
        "wx", "wy", "wz"
    ]

    state_units = [
        "m", "m", "m",
        "rad", "rad", "rad",
        "m/s", "m/s", "m/s",
        "rad/s", "rad/s", "rad/s"
    ]

    time = np.arange(len(y)) * dt

    fig, axes = plt.subplots(1, 3, figsize=(8, 4))
    axes = axes.flatten()

    for i in range(3):
        axes[i].plot(time, error[:, i], label=state_labels[i])

        axes[i].set_title(f"{state_labels[i]} error")
        axes[i].set_xlabel("Time [s]")
        axes[i].set_ylabel(f"Error [{state_units[i]}]")

        axes[i].axhline(0, color='black', linewidth=0.8)
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()
