import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time

from scipy.sparse.linalg import (
    gmres,
    LinearOperator,
    spsolve_triangular,
    spilu
)

# ============================================================
# 1. Problem parameters
# ============================================================

N = 40
epsi = 0.01
beta1 = 1.0
beta2 = 1.0

h = 1.0 / (N + 1)
n = N * N

# ============================================================
# 2. Build the matrix A and right-hand side b
# ============================================================

A = sp.lil_matrix((n, n), dtype=float)
b = np.zeros(n)

for j in range(1, N + 1):
    for i in range(1, N + 1):

        k = (j - 1) * N + (i - 1)

        x = i * h
        y = j * h

        b[k] = (
            2 * epsi * np.pi**2
            * np.sin(np.pi*x) * np.sin(np.pi*y)
            + np.pi * beta1
            * np.cos(np.pi*x) * np.sin(np.pi*y)
            + np.pi * beta2
            * np.sin(np.pi*x) * np.cos(np.pi*y)
        )

        A[k, k] = 4 * epsi / h**2

        if i < N:
            A[k, k + 1] = -epsi / h**2 + beta1 / (2*h)

        if i > 1:
            A[k, k - 1] = -epsi / h**2 - beta1 / (2*h)

        if j < N:
            A[k, k + N] = -epsi / h**2 + beta2 / (2*h)

        if j > 1:
            A[k, k - N] = -epsi / h**2 - beta2 / (2*h)

A = A.tocsr()
b_norm = np.linalg.norm(b)

# ============================================================
# 3. GMRES parameters
# ============================================================

x0 = np.zeros(n)
tol = 1e-8
restart = 20
maxit = 500

# ============================================================
# 4. Preconditioner construction
# ============================================================

# Jacobi
def build_jacobi():
    start = time.perf_counter()

    M_inv = sp.diags(1.0 / A.diagonal(), format='csr')
    M = LinearOperator(
        A.shape,
        matvec=lambda r: M_inv @ r,
        dtype=float
    )

    setup_time = time.perf_counter() - start
    return M, setup_time


# SSOR
def build_ssor():
    start = time.perf_counter()

    omega = 1.0

    D = sp.diags(A.diagonal(), format='csc')
    L = sp.tril(A, k=-1, format='csc')
    U = sp.triu(A, k=1, format='csc')

    B_lower = (D + omega * L).tocsc()
    B_upper = (D + omega * U).tocsc()

    d = A.diagonal()

    def apply_ssor(r):
        y = spsolve_triangular(
            B_lower, r, lower=True
        )

        z = spsolve_triangular(
            B_upper, d * y, lower=False
        )

        return omega * (2 - omega) * z

    M = LinearOperator(
        A.shape,
        matvec=apply_ssor,
        dtype=float
    )

    setup_time = time.perf_counter() - start
    return M, setup_time


# ILU(0)-like
def build_ilu():
    start = time.perf_counter()

    ilu = spilu(
        A.tocsc(),
        drop_tol=0.0,
        fill_factor=1.0,
        permc_spec='NATURAL',
        diag_pivot_thresh=0.0
    )

    M = LinearOperator(
        A.shape,
        matvec=ilu.solve,
        dtype=float
    )

    setup_time = time.perf_counter() - start
    return M, setup_time


# ============================================================
# 5. Run GMRES and record results
# ============================================================

results = []
histories = {}

def run_gmres(name, M=None, setup_time=0.0):

    residual_history = []

    def callback(residual):
        residual_history.append(residual)

    start = time.perf_counter()

    x, info = gmres(
        A, b,
        x0=x0,
        M=M,
        rtol=tol,
        atol=0.0,
        restart=restart,
        maxiter=int(np.ceil(maxit / restart)),
        callback=callback,
        callback_type='pr_norm'
    )

    solve_time = time.perf_counter() - start

    relative_residual = (
        np.linalg.norm(b - A @ x) / b_norm
    )

    total_time = setup_time + solve_time
    iterations = len(residual_history)

    status = "Converged" if info == 0 else "Not converged"
    results.append({
        "Method": name,
        "Status": status,
        "Iterations": iterations,
        "Residual": relative_residual,
        "Setup": setup_time,
        "Solve": solve_time,
        "Total": total_time
    })

    histories[name] = residual_history


# ============================================================
# 6. Run the four experiments
# ============================================================

# 1. GMRES without preconditioning
run_gmres("GMRES")

# 2. GMRES + Jacobi
M_jacobi, setup_jacobi = build_jacobi()
run_gmres("GMRES + Jacobi", M_jacobi, setup_jacobi)

# 3. GMRES + SSOR
M_ssor, setup_ssor = build_ssor()
run_gmres("GMRES + SSOR", M_ssor, setup_ssor)

# 4. GMRES + ILU
M_ilu, setup_ilu = build_ilu()
run_gmres("GMRES + ILU(0)", M_ilu, setup_ilu)


# ============================================================
# 7. Print comparison table
# ============================================================

print("\nGMRES PRECONDITIONING RESULTS")
print("-" * 105)

print(
    f"{'Method':<22}"
    f"{'Status':<15}"
    f"{'Iters':<9}"
    f"{'Residual':<15}"
    f"{'Setup(s)':<12}"
    f"{'Solve(s)':<12}"
    f"{'Total(s)':<12}"
)

print("-" * 105)

for r in results:
    print(
        f"{r['Method']:<22}"
        f"{r['Status']:<15}"
        f"{r['Iterations']:<9}"
        f"{r['Residual']:<15.4e}"
        f"{r['Setup']:<12.6f}"
        f"{r['Solve']:<12.6f}"
        f"{r['Total']:<12.6f}"
    )

print("-" * 105)


# ============================================================
# 8. Plot convergence curves
# ============================================================

plt.figure(figsize=(9, 6))

for name, history in histories.items():
    plt.semilogy(history, label=name)

plt.xlabel("Iteration")
plt.ylabel("Preconditioned Residual Norm")
plt.title("GMRES Convergence with Different Preconditioners")

plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("GMRES_Preconditioning_Convergence.png", dpi=300)
plt.show()
