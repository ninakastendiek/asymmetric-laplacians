
import numpy as np
import scipy
from julia import Main
Main.eval("using NumericalRange")

def julia_num_radius(A):
    Main.A = A
    # call nrange from NumericalRange.jl
    Main.eval("f, e = NumericalRange.nrange(A; thmax=100, noplot=true)")
    f = np.array(Main.f)  # boundary points

    numerical_radius = np.max(np.abs(f))
    return numerical_radius

def julia_phi(A):
    Main.A = A
    # call nrange from NumericalRange.jl
    Main.eval("f, e = NumericalRange.nrange(A; thmax=100, noplot=true)")
    f = np.array(Main.f)  # boundary points
    
    phis = np.angle(f)
    return float(phis.min()), float(phis.max())

def reduce(M):
    N = np.shape(M)[0]
    e = np.ones((N, 1))
    Q, _ = np.linalg.qr(np.hstack([e, np.eye(N)])) # QR decomposition for orthonormal matrix Q
    Q1 = Q[:, 1:] # select basis for subspace perp 1
    M1 = Q1.T @ M @ Q1
    return M1

def compute_xi0(L_plus_r, L_minus_r):
    
    w, U = np.linalg.eigh(L_plus_r)
    Q = U @ np.diag(1.0 / np.sqrt(w)) @ U.T
    M = Q @ L_minus_r @ Q
    M = 0.5 * (M +  M.T)
    eigvals = scipy.linalg.eigvalsh(M)
    
    xi0 = np.min(eigvals)

    return xi0

def compute_rho0(L_plus_r, L_minus_r):
    
    # Q = scipy.linalg.cholesky(L_plus_r, lower=True)
    # X = scipy.linalg.solve_triangular(Q, L_minus_r, lower=True)
    # Y = scipy.linalg.solve_triangular(Q, X.T, lower=True)
    # M = Y.T
    w, U = np.linalg.eigh(L_plus_r)
    Q = U @ np.diag(1.0 / np.sqrt(w)) @ U.T
    M = Q @ L_minus_r @ Q

    #rho0 = np.linalg.svd(M, compute_uv=False)[0]
    rho0 = julia_num_radius(M)

    return rho0

def compute_rho1(A_plus, A_minus, lambda_2):

    A_plus_min = np.min(A_plus[A_plus != 0])
    
    rho1 = max(np.abs(A_minus).sum(axis=1)) / (A_plus_min * lambda_2)

    return rho1

def compute_rho2(A_plus, A_minus, A_0, lambda_2n):
    
    d_norm = A_plus.sum(axis=1) / A_0.sum(axis=1)
    inv_d = 1.0 / d_norm  
    F = inv_d[:, None] + inv_d[None, :]
    A_minus_d = abs(A_minus)**2 * F
    A_plus_max = np.max(A_plus)
    A_plus_min = np.min(A_plus[A_plus != 0])
    A_minus_max = np.max(A_minus_d)
    
    rho2 = np.sqrt((A_plus_max*A_minus_max)/(lambda_2n*A_plus_min**2))
    
    return rho2









# def num_radius(A, n_theta=200):
#     w = 0.0
#     for k in range(n_theta):
#         theta = 2*np.pi*k/n_theta
#         H = 0.5*(np.exp(-1j*theta)*A + np.exp(1j*theta)*A.conj().T) 
#         #lam_max, _ = scipy.sparse.linalg.eigsh(H, k=1, which='LA')
#         #lam_max = np.real(lam_max[0])
#         lam_max = scipy.linalg.eigvalsh(H)[-1]
#         w = max(w, lam_max)
#     return w

# def num_range_boundary(A, thmax=200):
#     """
#     Approximate boundary points of the numerical range W(A)
#     """
#     A = np.asarray(A)
#     n = A.shape[0]
#     if A.shape != (n, n):
#         raise ValueError("A must be square")

#     A = A.astype(np.complex128, copy=False)
#     A_star = A.conj().T

#     thetas = np.linspace(0.0, 2*np.pi, thmax, endpoint=False)
#     f = np.empty(thmax, dtype=np.complex128)

#     for k, th in enumerate(thetas):
#         e = np.exp(-1j * th)

#         # H is Hermitian
#         H = 0.5 * (e * A + np.conj(e) * A_star)

#         # Largest eigenvector of H
#         vals, vecs = np.linalg.eigh(H)
#         v = vecs[:, -1]

#         # boundary point
#         f[k] = np.vdot(v, A @ v)

#     return f

# def phi_num_range(A, thmax=200):
#     f = num_range_boundary(A, thmax=thmax)
#     phis = np.angle(f) 
#     return float(phis.min()), float(phis.max())