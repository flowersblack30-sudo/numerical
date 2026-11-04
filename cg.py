import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import cg, spsolve
import time

# ============================================================
# 1. Parameters
# ============================================================

N = 40
h = 1.0 / (N + 1)
n = N * N
tol = 1e-10

# ============================================================
# 2. Construct the 2D Poisson matrix
# ============================================================

T = sp.diags(
    [-np.ones(N-1), 2*np.ones(N), -np.ones(N-1)],
    [-1, 0, 1],
    format='csr'
)

I = sp.eye(N, format='csr')

A = (sp.kron(I, T) + sp.kron(T, I)) / h**2
A = A.tocsr()

# Right-hand side
b = np.ones(n)

# Initial guess
x0 = np.zeros(n)

# ============================================================
# 3. Theoretical convergence bound
# ============================================================

# Exact condition number for the 2D discrete Poisson matrix
kappa = 1.0 / np.tan(np.pi / (2 * (N + 1)))**2

q = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)

print("2D Poisson Problem")
print("-------------------------")
print("Grid size:", N, "x", N)
print("Number of unknowns:", n)
print("Matrix shape:", A.shape)
print("Nonzero entries:", A.nnz)
print("Condition number:", kappa)
print("Convergence factor q:", q)

# ============================================================
# 4. Reference solution
# ============================================================

# Direct solution used only to evaluate the actual error
x_exact = spsolve(A, b)

# Initial energy error
e0 = x_exact - x0
initial_energy_error = np.sqrt(e0 @ (A @ e0))

# ============================================================
# 5. CG with history recording
# ============================================================

residual_vectors = []
energy_errors = []

# Record initial residual and error (iteration 0)
r0 = b - A @ x0
residual_vectors.append(r0.copy())
energy_errors.append(1.0)

def cg_callback(xk):

    # Residual vector at the current iteration
    rk = b - A @ xk
    residual_vectors.append(rk.copy())

    # Actual energy error
    ek = x_exact - xk
    error_energy = np.sqrt(ek @ (A @ ek))

    relative_energy_error = (
        error_energy / initial_energy_error
    )

    energy_errors.append(relative_energy_error)


start = time.perf_counter()

x_cg, info = cg(
    A, b,
    x0=x0,
    rtol=tol,
    atol=0.0,
    maxiter=10*n,
    callback=cg_callback
)

runtime = time.perf_counter() - start

iterations = len(residual_vectors) - 1

# ============================================================
# 6. Theoretical bound and actual error
# ============================================================

k_values = np.arange(iterations + 1)

theoretical_bound = 2 * q**k_values

# ============================================================
# 7. Loss of orthogonality
# ============================================================

R = np.array(residual_vectors)

norms = np.linalg.norm(R, axis=1)

# Normalized inner products between residual vectors
denominator = np.outer(norms, norms)

Gram = (R @ R.T) / denominator

# Avoid division issues if any residual is zero
Gram = np.nan_to_num(Gram)

# Maximum off-diagonal absolute inner product
orthogonality_loss = []

for k in range(1, len(R)):
    G = np.abs(Gram[:k+1, :k+1]).copy()
    np.fill_diagonal(G, 0.0)

    orthogonality_loss.append(np.max(G))

# ============================================================
# 8. Results
# ============================================================

relative_residual = (
    np.linalg.norm(b - A @ x_cg) / np.linalg.norm(b)
)

print("\nCG Results")
print("-------------------------")
print("Status:", "Converged" if info == 0 else "Not converged")
print("Iterations:", iterations)
print(f"Relative residual: {relative_residual:.6e}")
print(f"Runtime: {runtime:.6f} s")

print("\nOrthogonality")
print("-------------------------")
print(
    "Maximum loss of orthogonality:",
    f"{max(orthogonality_loss):.6e}"
    if orthogonality_loss else "N/A"
)
# ============================================================
# 9. Plot: theoretical bound vs actual convergence
# ============================================================

plt.figure(figsize=(9, 6))

plt.semilogy(
    k_values,
    energy_errors,
    'o-',
    markersize=3,
    label='Actual relative energy error'
)

plt.semilogy(
    k_values,
    theoretical_bound,
    '--',
    label='Theoretical convergence bound'
)

plt.xlabel("Iteration")
plt.ylabel("Relative energy error")
plt.title("CG: Theoretical Bound vs Actual Convergence")
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("CG_Theoretical_Bound.png", dpi=300)

# ============================================================
# 10. Plot: loss of orthogonality
# ============================================================

plt.figure(figsize=(9, 6))

plt.semilogy(
    range(1, len(orthogonality_loss) + 1),
    orthogonality_loss,
    'o-',
    markersize=3
)

plt.xlabel("Iteration")
plt.ylabel("Maximum normalized inner product")
plt.title("Loss of Orthogonality of CG Residuals")

plt.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("CG_Orthogonality_Loss.png", dpi=300)

# ============================================================
# 11. Plot: residual orthogonality matrix
# ============================================================

plt.figure(figsize=(7, 6))

plt.imshow(
    np.abs(Gram),
    cmap='hot',
    origin='lower',
    aspect='auto',
    vmin=0,
    vmax=1
)

plt.colorbar(label='Absolute normalized inner product')
plt.xlabel("Iteration")
plt.ylabel("Iteration")
plt.title("Residual Orthogonality Matrix")

plt.tight_layout()
plt.savefig("CG_Orthogonality_Matrix.png", dpi=300)

plt.show()
