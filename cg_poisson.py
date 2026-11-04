import numpy as np
import scipy.sparse as sp
import time
from scipy.sparse.linalg import LinearOperator, spsolve_triangular
import matplotlib.pyplot as plt
 
N = 40
h = 1 / (N + 1)
n = N * N

e = np.ones(n)
A = sp.diags(
    [-e, -e, 4*e, -e, -e],
    [-N, -1, 0, 1, N],
    shape=(n, n), format='lil'
)
for i in range(1, N):
    A[i*N, i*N - 1] = 0
    A[i*N - 1, i*N] = 0
A = (A / h**2).tocsr()

b = np.ones(n)
b_norm = np.linalg.norm(b)

print("Matrix size:", A.shape, "| nnz:", A.nnz)

 
def cg_manual(A, b, M=None, tol=1e-8, maxiter=None):
    """
    Conjugate Gradient with optional preconditioner M (LinearOperator or sparse).
    Returns x, info, iters, residual_history (relative ||r_k||/||b||).
    """
    n = b.size
    if maxiter is None:
        maxiter = 10 * n

    x = np.zeros(n)
    r = b.copy()

    if M is None:
        z = r.copy()
    else:
        z = M @ r

    p = z.copy()
    rz = r @ z

    res_hist = []
    info = 1  # not converged by default

    for k in range(maxiter):
        Ap = A @ p
        pAp = p @ Ap
        if pAp <= 0:
            info = 2  # breakdown
            break

        alpha = rz / pAp
        x += alpha * p
        r -= alpha * Ap

        relres = np.linalg.norm(r) / b_norm
        res_hist.append(relres)

        if relres < tol:
            info = 0
            break

        if M is None:
            z = r.copy()
        else:
            z = M @ r

        rz_new = r @ z
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new

    return x, info, len(res_hist), res_hist

 
results = []
histories = {}


def run(name, M, setup_time):
    t0 = time.perf_counter()
    x, info, iters, hist = cg_manual(A, b, M=M, tol=1e-8)
    solve_time = time.perf_counter() - t0

    res_final = np.linalg.norm(A @ x - b) / b_norm
    total = setup_time + solve_time

    print(f"\nCG + {name}")
    print("  status   :", "Converged" if info == 0 else f"Not converged (info={info})")
    print("  iters    :", iters)
    print("  relres   :", res_final)
    print("  setup    :", setup_time, "s")
    print("  solve    :", solve_time, "s")
    print("  total    :", total, "s")

    results.append((name, info, iters, res_final, setup_time, solve_time, total))
    histories[name] = hist


 
run("no preconditioning", M=None, setup_time=0.0)

 
t0 = time.perf_counter()
M_jacobi = sp.diags(1.0 / A.diagonal(), format='csr')
setup_jacobi = time.perf_counter() - t0
run("Jacobi", M=M_jacobi, setup_time=setup_jacobi)

 
t0 = time.perf_counter()
omega = 1.0
D = sp.diags(A.diagonal(), format='csc')
L = sp.tril(A, k=-1, format='csc')
U = sp.triu(A, k= 1, format='csc')
B  = (D + omega * L).tocsc()
BT = (D + omega * U).tocsc()
d  = A.diagonal()

def apply_ssor(r):
    y = spsolve_triangular(B,  r, lower=True)
    z = spsolve_triangular(BT, d * y, lower=False)
    return z / (omega * (2 - omega))

M_ssor = LinearOperator(A.shape, matvec=apply_ssor, dtype=float)
setup_ssor = time.perf_counter() - t0
run("SSOR", M=M_ssor, setup_time=setup_ssor)

# 4) IC(0)
def incomplete_cholesky(A):
    A = A.tocsr()
    n = A.shape[0]
    rows = [dict() for _ in range(n)]
    for i in range(n):
        for p in range(A.indptr[i], A.indptr[i+1]):
            j = A.indices[p]; aij = A.data[p]
            if j >= i: continue
            s = 0.0
            for k, lik in rows[i].items():
                if k < j and k in rows[j]:
                    s += lik * rows[j][k]
            rows[i][j] = (aij - s) / rows[j][j]
        diag = A[i, i] - sum(v**2 for v in rows[i].values())
        if diag <= 0:
            raise ValueError("IC(0) failed")
        rows[i][i] = np.sqrt(diag)
    ri, ci, vv = [], [], []
    for i in range(n):
        for j, v in rows[i].items():
            ri.append(i); ci.append(j); vv.append(v)
    return sp.csr_matrix((vv, (ri, ci)), shape=(n, n))

t0 = time.perf_counter()
L_ic  = incomplete_cholesky(A)
LT_ic = L_ic.T.tocsr()

def apply_ic(r):
    y = spsolve_triangular(L_ic,  r, lower=True)
    return spsolve_triangular(LT_ic, y, lower=False)

M_ic = LinearOperator(A.shape, matvec=apply_ic, dtype=float)
setup_ic = time.perf_counter() - t0
run("IC(0)", M=M_ic, setup_time=setup_ic)

 
print("\n" + "=" * 90)
print(f"{'Method':<25}{'Iters':>7}{'Relres':>14}"
      f"{'Setup(s)':>12}{'Solve(s)':>12}{'Total(s)':>12}")
print("-" * 90)
for name, info, iters, res, ts, tv, tt in results:
    tag = "" if info == 0 else " [!]"
    print(f"{name:<25}{iters:>7d}{res:>14.3e}"
          f"{ts:>12.4f}{tv:>12.4f}{tt:>12.4f}{tag}")
print("=" * 90)

 
plt.figure(figsize=(9, 6))

 
plt.semilogy(range(1, len(histories["no preconditioning"]) + 1), 
             histories["no preconditioning"], 
             label='CG', linestyle='--', linewidth=2.5, color='blue')

plt.semilogy(range(1, len(histories["Jacobi"]) + 1), 
             histories["Jacobi"], 
             label='CG + Jacobi', linewidth=1.5, color='orange')

plt.semilogy(range(1, len(histories["SSOR"]) + 1), 
             histories["SSOR"], 
             label='CG + SSOR', linewidth=1.5, color='green')

plt.semilogy(range(1, len(histories["IC(0)"]) + 1), 
             histories["IC(0)"], 
             label='CG + IC(0)', linewidth=1.5, color='red')

plt.xlabel('Iteration')
plt.ylabel(r'Relative residual \(\|r_k\|/\|b\|\)')
plt.title('Convergence History of CG with Different Preconditioners')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('CG_Convergence_Fixed.png', dpi=300)
plt.show()
