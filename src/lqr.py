import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
from scipy.linalg import expm



# discretization
# x_dot = A x + B u  →  x[k+1] = Ad x[k] + Bd u[k]
def discretize(A, B, dt):
    n = A.shape[0]
    m = B.shape[1]

    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B

    Md = expm(M * dt)

    Ad = Md[:n, :n]
    Bd = Md[:n, n:]

    return Ad, Bd


# discrete LQR func
def dlqr(A, B, Q, R):
    K, S, E = ctrl.dlqr(A, B, Q, R)
    return np.array(K), S, E

def compute_dlqr(Q, R, dt):
    # load data
    data = np.load('linearised_matrices.npz')
    A_all = data['A']
    B_all = data['B']

    n = A_all.shape[0]

    # compute gain schedule
    K_mat = []

    for k in range(n):
        A = A_all[k]
        B = B_all[k]

        Ad, Bd = discretize(A, B, dt)

        K, _, _ = dlqr(Ad, Bd, Q, R)
        K_mat.append(K)


    K_mat = np.array(K_mat)

    np.savez("K_matrices.npz", K=K_mat)

    print("K shape:", K_mat.shape)

    # stability check
    eig_max = []

    for k in range(n):
        Acl = A_all[k] - B_all[k] @ K_mat[k]
        eig = np.linalg.eigvals(Acl)
        eig_max.append(np.max(np.real(eig)))

    eig_max = np.array(eig_max)

    plt.figure()
    plt.plot(eig_max)
    plt.axhline(0, color='r')
    plt.title("Max real eigenvalue of (A - BK)")
    plt.grid()
    plt.show()


    # gain norm
    K_norm = np.linalg.norm(K_mat.reshape(n, -1), axis=1)

    plt.figure()
    plt.plot(K_norm)
    plt.grid()
    plt.title("||K|| (DLQR)")
    plt.show()

if __name__ == "__main__":
    compute_dlqr()
