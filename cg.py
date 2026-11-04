import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# ============================================================
# 1) Parameters
# ============================================================
EPSILON = 0.01
BETA_X  = 1.0
BETA_Y  = 1.0
N       = 50

# ============================================================
# 2) Exact solution, source term, and boundary conditions
# ============================================================
def exact_solution(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def source_term(x, y):
    sx = np.sin(np.pi * x)
    sy = np.sin(np.pi * y)
    cx = np.cos(np.pi * x)
    cy = np.cos(np.pi * y)

    diffusion    = 2.0 * EPSILON * np.pi**2 * sx * sy
    convection_x = BETA_X * np.pi * cx * sy
    convection_y = BETA_Y * np.pi * sx * cy

    return diffusion + convection_x + convection_y

def boundary_value(x, y):
    return 0.0

# ============================================================
# 3) Assemble A and b
# ============================================================
def assemble_system(n, epsilon, beta_x, beta_y, f, g):
    if n < 1:
        raise ValueError("n must be at least 1.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if beta_x < 0 or beta_y < 0:
        raise ValueError(
            "This implementation uses backward upwind "
            "and requires beta_x >= 0 and beta_y >= 0."
        )

    h = 1.0 / (n + 1)
    x = np.linspace(0.0, 1.0, n + 2)
    y = np.linspace(0.0, 1.0, n + 2)

    a_p = 4.0 * epsilon / h**2 + (beta_x + beta_y) / h
    a_w = -epsilon / h**2 - beta_x / h
    a_e = -epsilon / h**2
    a_s = -epsilon / h**2 - beta_y / h
    a_n = -epsilon / h**2

    A = lil_matrix((n * n, n * n), dtype=float)
    b = np.zeros(n * n, dtype=float)

    # x varies fastest: k = j*n + i
    for j in range(n):
        for i in range(n):
            k  = j * n + i
            xi = x[i + 1]
            yj = y[j + 1]

            A[k, k] = a_p
            b[k]    = f(xi, yj)

            # West
            if i > 0:
                A[k, k - 1] = a_w
            else:
                b[k] -= a_w * g(0.0, yj)

            # East
            if i < n - 1:
                A[k, k + 1] = a_e
            else:
                b[k] -= a_e * g(1.0, yj)

            # South
            if j > 0:
                A[k, k - n] = a_s
            else:
                b[k] -= a_s * g(xi, 0.0)

            # North
            if j < n - 1:
                A[k, k + n] = a_n
            else:
                b[k] -= a_n * g(xi, 1.0)

    coefficients = {
        "a_P": a_p,
        "a_W": a_w,
        "a_E": a_e,
        "a_S": a_s,
        "a_N": a_n,
    }

    return A.tocsr(), b, x, y, h, coefficients

# ============================================================
# 4) Solve, verify, and plot
# ============================================================
def main():
    A, b, x, y, h, coefficients = assemble_system(
        n=N,
        epsilon=EPSILON,
        beta_x=BETA_X,
        beta_y=BETA_Y,
        f=source_term,
        g=boundary_value,
    )

    u = spsolve(A, b)

    # U[j, i] corresponds to coordinates (x[i], y[j]).
    X, Y = np.meshgrid(x, y, indexing="xy")

    U = np.zeros_like(X)
    U[1:-1, 1:-1] = u.reshape((N, N))

    # Insert Dirichlet boundary values.
    U[0,  :] = [boundary_value(xi, 0.0) for xi in x]
    U[-1, :] = [boundary_value(xi, 1.0) for xi in x]
    U[:,  0] = [boundary_value(0.0, yj) for yj in y]
    U[:, -1] = [boundary_value(1.0, yj) for yj in y]

    U_exact = exact_solution(X, Y)
    error          = np.abs(U - U_exact)
    interior_error = U[1:-1, 1:-1] - U_exact[1:-1, 1:-1]

    residual        = b - A @ u
    residual_norm   = np.linalg.norm(residual)
    relative_residual = residual_norm / max(
        np.linalg.norm(b), np.finfo(float).tiny
    )

    max_error        = np.max(np.abs(interior_error))
    discrete_l2_error = h * np.linalg.norm(interior_error)

    asymmetry = A - A.T
    asymmetry.eliminate_zeros()

    print(f"Interior grid         : {N} x {N}")
    print(f"Number of unknowns    : {N * N}")
    print(f"h                     : {h:.10f}")
    print(f"Matrix shape          : {A.shape}")
    print(f"Nonzero entries       : {A.nnz}")
    print(f"Is symmetric?         : {asymmetry.nnz == 0}")

    print("\nStencil coefficients:")
    for name, value in coefficients.items():
        print(f"  {name} = {value:.8f}")

    print("\nVerification:")
    print(f"Residual norm         : {residual_norm:.6e}")
    print(f"Relative residual     : {relative_residual:.6e}")
    print(f"Maximum nodal error   : {max_error:.6e}")
    print(f"Discrete L2 error     : {discrete_l2_error:.6e}")

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 4.5), constrained_layout=True
    )

    panels = [
        (U,       "Numerical solution"),
        (U_exact, "Exact solution"),
        (error,   "Absolute error"),
    ]

    for ax, (values, title) in zip(axes, panels):
        image = ax.pcolormesh(
            X, Y, values, shading="auto", cmap="viridis"
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.colorbar(image, ax=ax)

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.spy(A, markersize=0.5)
    ax2.set_title("Sparsity pattern of A")
    ax2.set_xlabel("Column index")
    ax2.set_ylabel("Row index")
    fig2.tight_layout()

    plt.show()
 
if __name__ == "__main__":
    main()
