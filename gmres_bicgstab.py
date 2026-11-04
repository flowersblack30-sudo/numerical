import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import gmres, bicgstab


# ==================================================
# 1. Problem parameters
# ==================================================

N = 40
epsi = 0.01
beta1 = 1
beta2 = 1

h = 1 / (N + 1)
n = N * N


# ==================================================
# 2. Construct the linear system A*x = b
# ==================================================

A = lil_matrix((n, n))
b = np.zeros(n)
u_exact = np.zeros(n)

for j in range(1, N + 1):
    for i in range(1, N + 1):

        k = (j - 1) * N + (i - 1)

        x = i * h
        y = j * h

        # Exact solution
        u_exact[k] = np.sin(np.pi * x) * np.sin(np.pi * y)

        # Right-hand side
        b[k] = (
            2 * epsi * np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)
            + np.pi * beta1 * np.cos(np.pi*x) * np.sin(np.pi*y)
            + np.pi * beta2 * np.sin(np.pi*x) * np.cos(np.pi*y)
        )

        # Main diagonal
        A[k, k] = 4 * epsi / h**2

        # East
        if i < N:
            A[k, k + 1] = -epsi / h**2 + beta1 / (2 * h)

        # West
        if i > 1:
            A[k, k - 1] = -epsi / h**2 - beta1 / (2 * h)

        # North
        if j < N:
            A[k, k + N] = -epsi / h**2 + beta2 / (2 * h)

        # South
        if j > 1:
            A[k, k - N] = -epsi / h**2 - beta2 / (2 * h)

# Convert to CSR format for efficient iterative solving
A = A.tocsr()


# ==================================================
# 3. Initial conditions
# ==================================================

x0 = np.zeros(n)
tol = 1e-8
maxit = 500


# ==================================================
# 4. Influence of restart parameter m
# ==================================================

M = [10, 20, 40, 80]

print("\nGMRES RESULTS")

for m in M:

    residual_history = []

    def gmres_callback(residual):
        residual_history.append(residual)

    start = time.perf_counter()

    x, info = gmres(
        A, b,
        x0=x0,
        rtol=tol,
        atol=0.0,
        restart=m,
        maxiter=int(np.ceil(maxit / m)),
        callback=gmres_callback,
        callback_type='pr_norm'
    )

    runtime = time.perf_counter() - start

    total_iter = len(residual_history)
    residual = np.linalg.norm(b - A @ x) / np.linalg.norm(b)

    print(f"m = {m}")
    print(f"Iterations = {total_iter}")
    print(f"Runtime = {runtime:.6f} s")
    print(f"Residual = {residual:.4e}")
    print(f"Solver info = {info}\n")


# ==================================================
# 5. Comparison GMRES and BiCGSTAB
# ==================================================

# GMRES(20)

resg = []

def gmres_comparison_callback(residual):
    resg.append(residual)

start = time.perf_counter()

xg, flagg = gmres(
    A, b,
    x0=x0,
    rtol=tol,
    atol=0.0,
    restart=20,
    maxiter=int(np.ceil(maxit / 20)),
    callback=gmres_comparison_callback,
    callback_type='pr_norm'
)

time_gmres = time.perf_counter() - start
iter_gmres = len(resg)

residual_gmres = np.linalg.norm(b - A @ xg) / np.linalg.norm(b)


# BiCGSTAB

resb = []

def bicgstab_callback(xk):
    current_residual = np.linalg.norm(b - A @ xk) / np.linalg.norm(b)
    resb.append(current_residual)

start = time.perf_counter()

xb, flagb = bicgstab(
    A, b,
    x0=x0,
    rtol=tol,
    atol=0.0,
    maxiter=maxit,
    callback=bicgstab_callback
)

time_bicg = time.perf_counter() - start
iter_bicg = len(resb)

residual_bicg = np.linalg.norm(b - A @ xb) / np.linalg.norm(b)


print("\nCOMPARISON RESULTS")

print(f"GMRES iterations = {iter_gmres}")
print(f"GMRES time = {time_gmres:.6f} s")
print(f"GMRES residual = {residual_gmres:.4e}")
print(f"GMRES info = {flagg}")

print(f"\nBiCGSTAB iterations = {iter_bicg}")
print(f"BiCGSTAB time = {time_bicg:.6f} s")
print(f"BiCGSTAB residual = {residual_bicg:.4e}")
print(f"BiCGSTAB info = {flagb}")
# ==================================================
# 6. Plot exact and numerical solutions
# ==================================================

# MATLAB uses column-major ordering.
# Therefore, we use order='F' in NumPy.

U = xg.reshape((N, N), order='F')
Uexact = u_exact.reshape((N, N), order='F')

X, Y = np.meshgrid(
    np.arange(1, N + 1) * h,
    np.arange(1, N + 1) * h
)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, U.T, cmap='viridis')

ax.set_title('Numerical Solution using GMRES')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x,y)')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Uexact.T, cmap='viridis')

ax.set_title('Exact Solution')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x,y)')


# ==================================================
# 7. Plot residual comparison
# ==================================================

plt.figure()

plt.semilogy(resg, label='GMRES(20)')
plt.semilogy(resb, label='BiCGSTAB')

plt.xlabel('Iterations')
plt.ylabel('Relative residual')
plt.title('GMRES vs BiCGSTAB')

plt.legend()
plt.grid(True)

plt.show()