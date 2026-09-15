import numpy as np
import control as ctrl
import matplotlib.pyplot as plt

def compute_ctrl():
    data = np.load("linearised_matrices.npz")
    A = data['A']
    B = data['B']

    num = A.shape[0]

    rank = []

    state_labels = ["x", "y", "z", "theta", "psi", "phi", "xdot", "ydot", "zdot", "theta_dot", "psi_dot", "phi_dot"]

    n_states = len(state_labels)

    # store controllability contribution per state
    state_controllability = np.zeros((num, n_states))

    for k in range(num):

        C = ctrl.ctrb(A[k], B[k])
        r = np.linalg.matrix_rank(C)
        rank.append(r)

        U, S, Vh = np.linalg.svd(C)

        Uc = U[:, :r]

        P = Uc @ Uc.T

        state_controllability[k] = np.diag(P)

    rank = np.array(rank)


    bad_idx = np.where(rank < n_states)[0]

    if len(bad_idx) > 0:
        print("\nPoints with rank < full:")
        
    for k in bad_idx:
        print(f"\nk = {k}, rank = {rank[k]}")
        print("Most uncontrollable states:")

        sorted_states = np.argsort(state_controllability[k])

        for i in sorted_states[:5]:
            print(f"  {state_labels[i]} → {state_controllability[k,i]:.4f}")

    plt.figure(figsize=(14, 8))

    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.plot(state_controllability[:, i])
        plt.title(state_labels[i])
        plt.ylim(0.0, 1.2)
        plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compute_ctrl()
