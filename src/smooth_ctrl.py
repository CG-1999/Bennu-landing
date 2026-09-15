import numpy as np 
import control as ctrl 
import matplotlib.pyplot as plt 

data = np.load("linearised_matrices.npz") 
A = data['A'] 
B = data['B'] 

# check
if A.ndim == 2:
    A = np.expand_dims(A, axis=0)
    B = np.expand_dims(B, axis=0)

num = A.shape[0] 
n_states = A.shape[1] 
state_labels = ["x", "y", "z", "theta", "psi", "phi", "xdot", "ydot", "zdot", "theta_dot", "psi_dot", "phi_dot"] 

# Thresholds 
COND_THRESH = 1e12 
RANK_FULL = n_states 

# identify which indices are truly valid
valid_mask = np.zeros(num, dtype=bool)
rank_original = []

for k in range(num):
    C = ctrl.ctrb(A[k], B[k]) 
    r = np.linalg.matrix_rank(C) 
    cond = np.linalg.cond(C) 
    rank_original.append(r)
    
    if cond <= COND_THRESH and r == RANK_FULL:
        valid_mask[k] = True

# Safety check: what if every single matrix is bad?
if not np.any(valid_mask):
    raise ValueError("All matrices in the dataset are ill-conditioned! Cannot smooth.")

# Reconstruct clean arrays using the nearest valid neighbor
A_clean_list = []
B_clean_list = []
state_controllability = np.zeros((num, n_states)) 
valid_idx = np.where(valid_mask)[0]

for k in range(num):
    if valid_mask[k]:
       
        A_k = A[k]
        B_k = B[k]
    else:
        # Find the absolute nearest valid index (backward or forward)
        nearest_valid_k = valid_idx[np.argmin(np.abs(valid_idx - k))]
        A_k = A[nearest_valid_k]
        B_k = B[nearest_valid_k]
        print(f"Warning: k={k} is bad. Substituted with nearest valid k={nearest_valid_k}")

    A_clean_list.append(A_k)
    B_clean_list.append(B_k)
    
    # plotting
    C_clean = ctrl.ctrb(A_k, B_k) 
    U, S, Vh = np.linalg.svd(C_clean) 
    r_eff = np.linalg.matrix_rank(C_clean) 
    Uc = U[:, :r_eff] 
    P = Uc @ Uc.T 
    state_controllability[k] = np.diag(P) 

# Convert to final arrays
A_clean = np.array(A_clean_list) 
B_clean = np.array(B_clean_list) 
        
np.savez("ctrl_linearised_matrices.npz", A=A_clean, B=B_clean, valid_idx=valid_idx) 
print("\nSaved: ctrl_linearised_matrices.npz") 

# Plotting
plt.figure(figsize=(14, 8)) 
for i in range(n_states): 
    plt.subplot(3, 4, i + 1) 
    plt.plot(state_controllability[:, i]) 
    plt.title(state_labels[i]) 
    plt.ylim(0.0, 1.2)
    plt.grid(True) 

plt.tight_layout() 
plt.show()
