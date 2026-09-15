import sympy as sp
import numpy as np
from gravity_Intrep import GravityGrid

# Gravity model
gravity_model = GravityGrid("bennu_gravity_field30.npz")

# Symbolic variables
x, y, z, theta, psi, phi = sp.symbols('x y z theta psi phi', real=True)
dx, dy, dz, dtheta, dpsi, dphi = sp.symbols('dx dy dz dtheta dpsi dphi', real=True)
ddx, ddy, ddz, ddtheta, ddpsi, ddphi = sp.symbols('ddx ddy ddz ddtheta ddpsi ddphi', real=True)
m, Ixx, Iyy, Izz = sp.symbols('m Ixx Iyy Izz', positive=True)
T1, T2, T3, T4, T5, T6, T7, T8, T9, T10 = sp.symbols('T1:11', real=True)
g_x, g_y, g_z = sp.symbols('g_x g_y g_z', real=True)

a = 0.6  # arm length [m]

q = sp.Matrix([x, y, z, theta, psi, phi])
q_dot = sp.Matrix([dx, dy, dz, dtheta, dpsi, dphi])
q_ddot = sp.Matrix([ddx, ddy, ddz, ddtheta, ddpsi, ddphi])

# Rotation matrices
Rz = sp.Matrix([
    [sp.cos(phi), -sp.sin(phi), 0],
    [sp.sin(phi), sp.cos(phi), 0],
    [0, 0, 1],
])
Ry = sp.Matrix([
    [ sp.cos(psi), 0, sp.sin(psi)],
    [ 0, 1, 0],
    [-sp.sin(psi), 0, sp.cos(psi)],
])
Rx = sp.Matrix([
    [1, 0, 0],
    [0, sp.cos(theta), -sp.sin(theta)],
    [0, sp.sin(theta), sp.cos(theta)],
])

S_PK = Rx * Ry * Rz  

# Inertia tensor in world frame
I_k = sp.diag(Ixx, Iyy, Izz)
I = sp.simplify(S_PK * I_k * S_PK.T)


# Translational Jacobian  J_T  (3×6)
J_T = sp.Matrix.hstack(sp.eye(3), sp.zeros(3, 3))

# Rotational Jacobian  J_R  (3×6)
# vee() extracts the axial vector from a skew-symmetric matrix G:
#   G = [[  0, -w3,  w2],
#        [ w3,   0, -w1],
#        [-w2,  w1,   0]]
# so vee(G) = [w1, w2, w3] = [G[2,1], G[0,2], G[1,0]] 

def vee(G):
    return sp.Matrix([G[2, 1], G[0, 2], G[1, 0]])

G_theta = sp.diff(S_PK, theta) * S_PK.T
G_psi = sp.diff(S_PK, psi) * S_PK.T
G_phi = sp.diff(S_PK, phi) * S_PK.T

omega_theta = sp.simplify(vee(G_theta)) # 3×1
omega_psi = sp.simplify(vee(G_psi))   # 3×1
omega_phi = sp.simplify(vee(G_phi)) # 3×1

# J_R
J_R = sp.Matrix.hstack(sp.zeros(3, 3), omega_theta, omega_psi, omega_phi)

# Velocities
v = J_T * q_dot   # 3×1 linear velocity
omega = J_R * q_dot   # 3×1 angular velocity

# Kinetic energy
T_trans = sp.Rational(1, 2) * m * (v.T * v)[0]
T_rot = sp.Rational(1, 2) * (omega.T * I * omega)[0]
T = sp.simplify(T_trans + T_rot)

# Generalised forces
f_thrust = sp.Matrix([
    T4 + T7 - T5 - T2,
    T3 + T6 - T8 - T1,
    T9 - T10,
])

f_gravity = sp.Matrix([m * g_x, m * g_y, m * g_z])

# Torques in body frame
tau = sp.Matrix([
    ((T1 + T6) - (T3 + T8)) * (a / 2),
    ((T4 + T5) - (T2 + T7)) * (a / 2),
    ((T5 + T8 + T7 + T6) - (T1 + T4 + T3 + T2)) * (a / 2),
])

# Generalised thrust & gravity forces
d_t = J_T.T * (S_PK * f_thrust) + J_R.T * tau 
d_g = J_T.T * f_gravity

def build_symbolic_model():
    # Euler-Lagrange equations
    # Lag[i] = d/dt(∂T/∂q̇ᵢ) - ∂T/∂qᵢ 
    print("Building Euler-Lagrange terms …")
    Lag = sp.zeros(6, 1)
    for i in range(6):
        dT_dqdot_i = sp.diff(T, q_dot[i])
        dt_term = sum(
            sp.diff(dT_dqdot_i, q[j])     * q_dot[j]
            + sp.diff(dT_dqdot_i, q_dot[j]) * q_ddot[j]
            for j in range(6)
        )
        dT_dq_i = sp.diff(T, q[i])
        Lag[i] = sp.simplify(dt_term - dT_dq_i - d_g[i])

    # Mass matrix  M(q)  and Coriolis/centrifugal vector  g(q, q̇)
    print("Extracting mass matrix …")

    M_sym = Lag.jacobian(q_ddot)
    g_sym = Lag - M_sym*q_ddot

    # Nonlinear right-hand side:  q̈ = M⁻¹ (d_t − g_sym)
    print("Solving for f_nl (M \\ rhs) …")
    rhs = d_t - g_sym
    f_nl = M_sym.LUsolve(rhs)

    # State-space form
    X = sp.Matrix.vstack(q, q_dot)          # (12, 1)
    Xdot = sp.Matrix.vstack(q_dot, f_nl)       # (12, 1)
    U = sp.Matrix([T1, T2, T3, T4, T5, T6, T7, T8, T9, T10])

    print("X shape:", X.shape, "  Xdot shape:", Xdot.shape)

    # Symbolic Jacobians  A(q,q_dot,u)  and  B(q,q_dot,u)
    print("Computing A_sym Jacobian …")
    A_sym = Xdot.jacobian(X)
    print("Computing B_sym Jacobian …")
    B_sym = Xdot.jacobian(U)

    arg_syms = [
        x, y, z, theta, psi, phi,
        dx, dy, dz, dtheta, dpsi, dphi,
        g_x, g_y, g_z,
        T1, T2, T3, T4, T5, T6, T7, T8, T9, T10,
        Ixx, Iyy, Izz, m,
    ]

    A_func = sp.lambdify(arg_syms, A_sym, modules='numpy')
    B_func = sp.lambdify(arg_syms, B_sym, modules='numpy')

    return A_func, B_func

def compute_linearized_matrices(mass_val, I):

    A_func, B_func = build_symbolic_model()

    # Load trajectory data      
    data_traj_pva  = np.load('Nominal_pos_vel_acc_values.npz')
    r_t = data_traj_pva['r_t']      # (N, 3)
    r_dot = data_traj_pva['r_dot']    # (N, 3)

    g_field = gravity_model.evaluate(r_t) # (N, 3)

    data_traj_oaa  = np.load('Nominal_euler_angle_values.npz')
    E = data_traj_oaa['E']  # (N, 3)
    E_dot = data_traj_oaa['E_dot']  # (N, 3)

    data_traj_ctrl = np.load('Nominal_control_input.npz')
    u_r = data_traj_ctrl['u_r']     # (N, 10)


    N = len(r_t)
    A = np.zeros((N, 12, 12))
    B = np.zeros((N, 12, 10))
    Ixx_val = I[0,0]
    Iyy_val = I[1,1]
    Izz_val = I[2,2]
    mass = mass_val

    print(f"Evaluating {N} linearisation points …")
    for k in range(N):
        args = (
            r_t[k, 0], r_t[k, 1], r_t[k, 2],
            E[k, 0],   E[k, 1],   E[k, 2],
            r_dot[k, 0], r_dot[k, 1], r_dot[k, 2],
            E_dot[k, 0], E_dot[k, 1], E_dot[k, 2],
            g_field[k, 0], g_field[k, 1], g_field[k, 2],
            u_r[k, 0], u_r[k, 1], u_r[k, 2], u_r[k, 3], u_r[k, 4],
            u_r[k, 5], u_r[k, 6], u_r[k, 7], u_r[k, 8], u_r[k, 9],
            Ixx_val, Iyy_val, Izz_val, mass)

        A[k] = np.array(A_func(*args), dtype=float)
        B[k] = np.array(B_func(*args), dtype=float)

    print("Done.  A shape:", A.shape, "  B shape:", B.shape)
    np.savez('linearised_matrices.npz', A=A, B=B)
    print("Saved to linearised_matrices.npz")

if __name__ == "__main__":
    compute_linearized_matrices() 
