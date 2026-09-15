import sympy as sp
import numpy as np
from gravity_Intrep import GravityGrid

gravity_model = GravityGrid("bennu_gravity_field30.npz")

# Generalized coordinates & their derivatives
x, y, z, theta, psi, phi = sp.symbols('x y z theta psi phi', real=True)

dx, dy, dz, dtheta, dpsi, dphi = sp.symbols('dx dy dz dtheta dpsi dphi', real=True)
ddx, ddy, ddz, ddtheta, ddpsi, ddphi = sp.symbols('ddx ddy ddz ddtheta ddpsi ddphi', real=True)
m, Ixx, Iyy, Izz = sp.symbols('m Ixx Iyy Izz', real=True)
T1, T2, T3, T4, T5, T6, T7, T8, T9, T10 = sp.symbols('T1:11', real=True)
g_x, g_y, g_z = sp.symbols('g_x g_y g_z', real=True)

a = 0.6  # arm length [m]

q      = sp.Matrix([x,   y,   z,   theta,   psi,   phi])
q_dot  = sp.Matrix([dx,  dy,  dz,  dtheta,  dpsi,  dphi])
q_ddot = sp.Matrix([ddx, ddy, ddz, ddtheta, ddpsi, ddphi])

# Rotation matrices
Rz = sp.Matrix([
    [sp.cos(phi), -sp.sin(phi), 0],
    [sp.sin(phi),  sp.cos(phi), 0],
    [0, 0, 1]
])
Ry = sp.Matrix([
    [ sp.cos(psi), 0, sp.sin(psi)],
    [0,            1, 0           ],
    [-sp.sin(psi), 0, sp.cos(psi)]
])
Rx = sp.Matrix([
    [1, 0,              0             ],
    [0, sp.cos(theta), -sp.sin(theta) ],
    [0, sp.sin(theta),  sp.cos(theta) ]
])

S_PK = Rx * Ry * Rz  

# Inertia tensor in world frame
I_k = sp.diag(Ixx, Iyy, Izz)
I   = sp.simplify(S_PK * I_k * S_PK.T)

# Jacobians 
J_T = sp.Matrix.hstack(sp.eye(3), sp.zeros(3, 3))   # translational (3×6)

def vee(G):
    return sp.Matrix([G[2, 1], G[0, 2], G[1, 0]])

G_theta = sp.simplify(sp.diff(S_PK, theta) * S_PK.T)
G_psi = sp.simplify(sp.diff(S_PK, psi)   * S_PK.T)
G_phi = sp.simplify(sp.diff(S_PK, phi)   * S_PK.T)

omega_theta = vee(G_theta)
omega_psi = vee(G_psi)
omega_phi = vee(G_phi)

J_R = sp.Matrix.hstack(sp.zeros(3, 3),omega_theta,omega_psi,omega_phi) # rotational Jacobian (3×6)

# Velocities
v = J_T * q_dot          # linear velocity  (3×1)
omega = J_R * q_dot          # angular velocity (3×1)

# Kinetic energy
T_trans = sp.Rational(1, 2) * m * (v.T * v)[0]
T_rot = sp.Rational(1, 2) * (omega.T * I * omega)[0]
T = sp.simplify(T_trans + T_rot)

# Generalised forces
# Thrust forces in body frame
f_thrust = sp.Matrix([
    T4 + T7 - T5 - T2,
    T3 + T6 - T8 - T1,
    T9 - T10,
])

# Gravity force in world frame
f_gravity = sp.Matrix([m * g_x, m * g_y, m * g_z])

# Torques in body frame
tau = sp.Matrix([
    ((T1 + T6) - (T3 + T8)) * (a / 2),
    ((T4 + T5) - (T2 + T7)) * (a / 2),
    ((T5 + T8 + T7 + T6) - (T1 + T4 + T3 + T2)) * (a / 2),
])

# Generalised thrust force 
d_t = J_T.T * (S_PK * f_thrust) + J_R.T * tau 

# Generalised gravity force
d_g = J_T.T * f_gravity

# Euler-Lagrange: M(q)·q̈ + g(q,q̇) = d_t
# Lag[i] = d/dt(∂T/∂q̇ᵢ) - ∂T/∂qᵢ - Q_gravity[i]
Lag = sp.zeros(6, 1)

for i in range(6):
    dT_dqdot = sp.diff(T, q_dot[i])

    dt_term = 0
    for j in range(6):
        dt_term += sp.diff(dT_dqdot, q[j])      * q_dot[j]
        dt_term += sp.diff(dT_dqdot, q_dot[j])  * q_ddot[j]

    dT_dq = sp.diff(T, q[i])

    Lag[i] = sp.simplify(dt_term - dT_dq - d_g[i])

# Mass matrix M(q) 
n = len(q)
M_sym = sp.zeros(n)
for i in range(n):
    for j in range(n):
        M_sym[i, j] = sp.simplify(sp.diff(sp.diff(T, q_dot[i]), q_dot[j]))

# Coriolis/centrifugal + potential term:  g_sym = Lag - M·q̈
g_sym = sp.simplify(Lag.subs({ddx:0, ddy:0, ddz:0, ddtheta:0, ddpsi:0, ddphi:0}))

def build_B(B_sym, th, ps, ph):
        subs_B = {theta: th, psi: ps, phi: ph}
        return np.array(B_sym.subs(subs_B), dtype=float)

def build_B_symbolic():
        T_vec = sp.Matrix([T1, T2, T3, T4, T5, T6, T7, T8, T9, T10])
        cols = []
        for Ti in T_vec:
            col = sp.diff(d_t, Ti) 
            cols.append(col)
        return sp.Matrix.hstack(*cols)

 
def compute_nominal_control(mass_val, I):
    # Load nominal translational trajectory
    data_traj_pva = np.load('Nominal_pos_vel_acc_values.npz')
    r_t    = data_traj_pva['r_t']     # (N, 3)
    r_dot  = data_traj_pva['r_dot']   # (N, 3)
    r_ddot = data_traj_pva['r_ddot']  # (N, 3)

    # Load gravity field along the path
    g_field = gravity_model.evaluate(r_t) 

    # Load nominal Euler-angle trajectory 
    data_traj_oaa = np.load('Nominal_euler_angle_values.npz')
    E      = data_traj_oaa['E']       # (N, 3)  
    E_dot  = data_traj_oaa['E_dot']   # (N, 3)
    E_ddot = data_traj_oaa['E_ddot']  # (N, 3)

    theta_r = E[:, 0];  psi_r = E[:, 1];  phi_r = E[:, 2]
    theta_dot = E_dot[:, 0]; psi_dot = E_dot[:, 1]; phi_dot = E_dot[:, 2]
    theta_ddot = E_ddot[:, 0]; psi_ddot = E_ddot[:, 1]; phi_ddot = E_ddot[:, 2]

    qr = np.column_stack([r_t,    theta_r,    psi_r,    phi_r])
    qdot_r = np.column_stack([r_dot,  theta_dot,  psi_dot,  phi_dot])
    qddot_r= np.column_stack([r_ddot, theta_ddot, psi_ddot, phi_ddot])

    Ixx_val = I[0,0]
    Iyy_val = I[1,1]
    Izz_val = I[2,2]
    mass    = mass_val

    N = len(r_t)

    # Evaluate required generalised force d_r = M(q)·q̈ + g(q,q̇) 
    d_r_all = []

    for k in range(N):
        subs = {
            x: qr[k, 0], y: qr[k, 1], z: qr[k, 2], theta: qr[k, 3], psi: qr[k, 4], phi: qr[k, 5],

            dx: qdot_r[k, 0], dy: qdot_r[k, 1], dz: qdot_r[k, 2], dtheta: qdot_r[k, 3], dpsi: qdot_r[k, 4], dphi: qdot_r[k, 5],

            Ixx: Ixx_val, Iyy: Iyy_val, Izz: Izz_val, m: mass,

            g_x: g_field[k, 0], g_y: g_field[k, 1], g_z: g_field[k, 2],
        }

        M_num = np.array(M_sym.subs(subs), dtype=float)   # (6,6)
        g_num = np.array(g_sym.subs(subs), dtype=float)   # (6,1)

        qdd = qddot_r[k].reshape(6, 1)

        d_r = M_num @ qdd + g_num                          # (6,1)
        d_r_all.append(d_r)

    d_r_all = np.hstack(d_r_all).T    # (N, 6)

    # B matrix

       # (6×10)

    B_sym = build_B_symbolic()

    # Compute thrust inputs via regularised pseudo-inverse 
    u_all = []
    lam   = 1e-4   # Tikhonov regularisation

    for k in range(N):
        Bk = build_B(B_sym, theta_r[k], psi_r[k], phi_r[k])   # (6, 10)

        # Least-norm solution:  u = B^T (B B^T + λI)^{-1} d_r
        # (equivalent to right pseudo-inverse when B has full row rank)
        BBt = Bk @ Bk.T                                     
        Bk_pinv = Bk.T @ np.linalg.inv(BBt + lam * np.eye(6))  # (10, 6)

        dr = d_r_all[k].reshape(6, 1)
        u  = Bk_pinv @ dr                                # (10, 1)

        u_all.append(u)

    u_r = np.hstack(u_all).T    # (N, 10)
    np.savez('Nominal_control_input', u_r = u_r)

if __name__ == "__main__":
    compute_nominal_control()
