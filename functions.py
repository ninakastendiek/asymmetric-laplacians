
import numpy as np
import scipy
from dataclasses import dataclass, field
from typing import Optional
import matplotlib.pyplot as plt
import networkx as nx
from juliacall import Main as jl
jl.seval("using NumericalRange")

def julia_num_radius(A):
    f, e = jl.NumericalRange.nrange(A, thmax=100, noplot=True)
    f = np.array(f, dtype=np.complex128)
    return np.max(np.abs(f))

def julia_phi(A):
    f, e = jl.NumericalRange.nrange(A, thmax=100, noplot=True)
    f = np.array(f, dtype=np.complex128)
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

@dataclass
class ResultsSimple:
    alphas: list = field(default_factory=list)
    xi0s: list = field(default_factory=list)
    xi1s: list = field(default_factory=list)
    rho0s: list = field(default_factory=list)
    rho1s: list = field(default_factory=list)
    rho2s: list = field(default_factory=list)

    J_alpha = None
    Jp_alpha = None
    xi0_alpha = None
    xi1_alpha = None
    rho0_alpha = None
    rho1_alpha = None
    rho2_alpha = None
    neg_weight_alpha = None
    lambda2_L0 = None
    lambda2_L0n = None

@dataclass
class ResultsSecondOrder:
    alphas: list = field(default_factory=list)
    phi_L: list = field(default_factory=list)
    phi_rho0: list = field(default_factory=list)
    phi_rho1: list = field(default_factory=list)
    phi_rho2: list = field(default_factory=list)
    phi_zeta: list = field(default_factory=list)
    skar: list = field(default_factory=list)

    J_alpha = None
    phi_L_alpha = None
    phi_rho0_alpha = None
    phi_rho1_alpha = None
    phi_rho2_alpha = None
    phi_zeta_alpha = None
    skar_alpha = None
    lambda2_L0 = None
    lambda2_L0n = None

def compute_simple(G, N, alphas, seed=12):
    rng = np.random.default_rng(seed)
    res = ResultsSimple()

    A_0 = nx.to_numpy_array(G)

    # Symmetric part
    w_min = 0.5
    w_max = 1.5
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            wundirect = rng.uniform(w_min, w_max)
            W[i, j] = wundirect
            W[j, i] = wundirect

    # Antisymmetric part
    Delta = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            direction = rng.choice([-1, 1])
            wdirect = rng.uniform(w_min, w_max)
            Delta[i, j] = direction * wdirect
            Delta[j, i] = -direction * wdirect

    # Topology
    d_0 = A_0.sum(axis=1)
    L_0 = np.diag(d_0) - A_0
    L_0_norm = np.diag(d_0**(-0.5)) @ L_0 @ np.diag(d_0**(-0.5))
    res.lambda2_L0n = np.real(scipy.linalg.eigh(L_0_norm, eigvals_only=True, subset_by_index=[1, 1]))[0]
    res.lambda2_L0 = np.real(scipy.linalg.eigh(L_0, eigvals_only=True, subset_by_index=[1, 1]))[0]

    # # Projector
    # e = np.ones((N, 1))
    # Proj = np.eye(N) - (1/N) * (e @ e.T)

    res.alphas.append(0.0)
    res.xi0s.append(0.0)
    res.xi1s.append(0.0)
    res.rho0s.append(0.0)
    res.rho1s.append(0.0)
    res.rho2s.append(0.0)
    #for alpha in alphas:
    for i in range(1, len(alphas)):
        alpha_prev = alphas[i - 1]
        alpha = alphas[i]
        res.alphas.append(alpha)

        # Laplacian
        A = W + alpha * Delta
        A[~A_0.astype(bool)] = 0
        D = np.diag(A.sum(axis=1))
        L = D - A

        # Edge weights
        if res.neg_weight_alpha == None and (A < 0).any():
            res.neg_weight_alpha = alpha_prev
        dout_din = (A.sum(axis=1) - A.sum(axis=0))

        # Jacobian max eigenvalue
        eigvals_J = scipy.linalg.eigvals(-L)
        lambda_J = np.max(np.real(eigvals_J))
        #lambda_J, _ = scipy.sparse.linalg.eigs(-L, k=1, which='LR')
        #lambda_J = np.real(lambda_J[0])
        if res.J_alpha == None and lambda_J > 1e-10: 
            res.J_alpha = alpha_prev

        # # Projected Jacobian
        # L_perp = Proj @ L
        # #L_perp = reduce(L)
        # H_perp = 0.5 * (L_perp + L_perp.T)
        # eigvals_Jp = scipy.linalg.eigvalsh(-H_perp)
        # lambda_Jp = np.max(eigvals_Jp)
        # #lambda_Jp, _ = scipy.sparse.linalg.eigsh(-H_perp, k=1, which='LA')
        # #lambda_Jp = np.real(lambda_Jp[0])
        # if res.Jp_alpha == None and lambda_Jp > 1e-10: 
        #     res.Jp_alpha = alpha_prev

        # Undirected part
        A_plus = 0.5 * (A + A.T)
        L_plus = np.diag(A_plus.sum(axis=1)) - A_plus
        L_plus_r = reduce(L_plus)

        # Superdirected part
        A_minus = 0.5 * (A - A.T)
        L_minus = np.diag(A_minus.sum(axis=1)) - A_minus
        L_minus_r = reduce(L_minus)

        # xi0
        if res.xi0_alpha is None:
            xi0 = -compute_xi0(L_plus_r, L_minus_r)
            if xi0 < 1:
                res.xi0s.append(xi0)
            else: 
                res.xi0_alpha = alpha_prev
                res.xi0s.append(np.nan)
        else: 
            res.xi0s.append(np.nan)

        # xi1 
        if res.xi1_alpha is None:
            A_plus_min = np.min(A_plus[A_plus != 0])
            xi1 = -min(0.5*dout_din) / (A_plus_min * res.lambda2_L0)
            if xi1 < 1: 
                res.xi1s.append(xi1)
            else:
                res.xi1_alpha = alpha_prev
                res.xi1s.append(np.nan)
        else:
            res.xi1s.append(np.nan)

        # rho0
        if res.rho0_alpha is None:
            rho0 = compute_rho0(L_plus_r, L_minus_r)
            if rho0 < 1:
                res.rho0s.append(rho0)
            else: 
                res.rho0_alpha = alpha_prev
                res.rho0s.append(np.nan)
        else:
            res.rho0s.append(np.nan)

        # rho1
        if res.rho1_alpha is None:
            rho1 = compute_rho1(A_plus, A_minus, res.lambda2_L0)
            if rho1 < 1:
                res.rho1s.append(rho1)
            else: 
                res.rho1_alpha = alpha_prev
                res.rho1s.append(np.nan)
        else:
            res.rho1s.append(np.nan)

        # rho2
        if res.rho2_alpha is None:
            rho2 = compute_rho2(A_plus, A_minus, A_0, res.lambda2_L0n)
            if rho2 < 1:
                res.rho2s.append(rho2)
            else: 
                res.rho2_alpha = alpha_prev
                res.rho2s.append(np.nan)
        else:
            res.rho2s.append(np.nan)    

        if res.J_alpha is not None:
            break

    return res

def jacobian_second_order(m, gamma, L):

    N = L.shape[0]
    size = 2*N
    J = np.zeros((size, size), dtype=float)

    for n in range(N):
        J[n, N+n] = 1
    
    for n1 in range(N):
        for n2 in range(N):
            J[N+n1, n2] = -1/m[n1] * L[n1, n2]

    for n in range(N):
        J[N+n, N+n] = -gamma[n] / m[n]
            
    return J

def mixed_condition_second_order(gamma, m, gamma_m_ratio, phi2, sigma2_max):

    omega_c = gamma_m_ratio * np.tan(np.pi/2 - phi2)

    sigma1_max = np.max(1 / np.sqrt(gamma**2 + (m * omega_c)**2))

    cond = (1/omega_c) * sigma1_max * sigma2_max < 1

    return cond

def compute_second_order(G, N, alphas, seed=12):
    rng = np.random.default_rng(seed)
    res = ResultsSecondOrder()

    A_0 = nx.to_numpy_array(G)
    
    # Symmetric part
    w_min = 0.5
    w_max = 1.5
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            undirect = rng.uniform(w_min, w_max)
            W[i, j] = undirect
            W[j, i] = undirect

    # Antisymmetric part
    Delta = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            direct = rng.choice([-1, 1])
            wdirect = rng.uniform(w_min, w_max)
            Delta[i, j] = direct * wdirect
            Delta[j, i] = -direct * wdirect

    # Topology
    d_0 = A_0.sum(axis=1)
    L_0 = np.diag(d_0) - A_0
    L_0_norm = np.diag(d_0**(-0.5)) @ L_0 @ np.diag(d_0**(-0.5))
    res.lambda2_L0n =  np.real(scipy.linalg.eigh(L_0_norm, eigvals_only=True, subset_by_index=[1, 1]))[0]
    res.lambda2_L0 =  np.real(scipy.linalg.eigh(L_0, eigvals_only=True, subset_by_index=[1, 1]))[0]
    max_deg = max(dict(G.degree()).values())
    mean_deg = sum(dict(G.degree()).values()) / G.number_of_nodes()

    # Nodes
    gamma_m_ratio = 2 * np.sqrt(mean_deg)
    m_min, m_max = 0.5, 1.5
    m = rng.uniform(m_min, m_max, size=N)
    gamma = gamma_m_ratio * m

    res.alphas.append(0.0)
    res.phi_L.append(0.0)
    res.phi_rho0.append(0.0)
    res.phi_rho1.append(0.0)
    res.phi_rho2.append(0.0)
    res.phi_zeta.append(0.0)

    for i in range(1, len(alphas)):
        alpha_prev = alphas[i - 1]
        alpha = alphas[i]
        res.alphas.append(alpha)

        # Laplacian
        A = W + alpha * Delta
        A[~A_0.astype(bool)] = 0
        D = np.diag(A.sum(axis=1))
        L = D - A

        # Laplacian norm
        L_norm = scipy.linalg.svdvals(L)[0]
        edge_gains = np.sqrt(2 * (A**2 + A.T**2)) 
        np.fill_diagonal(edge_gains, 0.0)
        L_norm_approx = edge_gains.max() * max_deg
        dout_din = (A.sum(axis=1) - A.sum(axis=0))

        # Jacobian max eigenvalue
        J = jacobian_second_order(m, gamma, L)
        eigvals_J = np.real(scipy.linalg.eigvals(J))
        lambda_J = np.max(np.real(eigvals_J))
        if lambda_J > 1e-10: 
            res.J_alpha = alpha_prev

        # Undirected part
        A_plus = 0.5 * (A + A.T)
        L_plus = np.diag(A_plus.sum(axis=1)) - A_plus
        L_plus_r = reduce(L_plus)
    
        # Directed part
        A_minus = 0.5 * (A - A.T)
        L_minus = np.diag(A_minus.sum(axis=1)) - A_minus
        L_minus_r = reduce(L_minus)

        # phi L
        phi_L_value = np.nan
        if res.phi_L_alpha is None:
            xi0 = compute_xi0(L_plus_r, L_minus_r)
            if xi0 > -1:
                _, phi2 = julia_phi(reduce(L))
                cond = mixed_condition_second_order(gamma, m, gamma_m_ratio, phi2, sigma2_max=L_norm) 
                if cond: 
                    phi_L_value = phi2
                else:
                    res.phi_L_alpha = alpha_prev
            else: 
                res.phi_L_alpha = alpha_prev
        res.phi_L.append(phi_L_value)

        # rho0
        phi_rho0_alpha_value = np.nan
        if res.phi_rho0_alpha is None:
            rho0 = compute_rho0(L_plus_r, L_minus_r)
            if rho0 < 1:
                phi2 = np.arcsin(rho0)
                cond = mixed_condition_second_order(gamma, m, gamma_m_ratio, phi2, sigma2_max=L_norm) 
                if cond: 
                    phi_rho0_alpha_value = phi2
                else: 
                    res.phi_rho0_alpha = alpha_prev
            else: 
                res.phi_rho0_alpha = alpha_prev
        res.phi_rho0.append(phi_rho0_alpha_value)
        
        # rho1
        phi_rho1_alpha_value = np.nan
        if res.phi_rho1_alpha is None:
            rho1 = compute_rho1(A_plus, A_minus, res.lambda2_L0)
            if rho1 < 1:
                phi2 = np.arcsin(rho1)
                cond = mixed_condition_second_order(gamma, m, gamma_m_ratio, phi2, sigma2_max=L_norm_approx) 
                if cond: 
                    phi_rho1_alpha_value = phi2
                else:
                    res.phi_rho1_alpha = alpha_prev
            else: 
                res.phi_rho1_alpha = alpha_prev
        res.phi_rho1.append(phi_rho1_alpha_value)

        # rho2
        phi_rho2_alpha_value = np.nan
        if res.phi_rho2_alpha is None: 
            rho2 = compute_rho2(A_plus, A_minus, A_0, res.lambda2_L0n)
            if rho2 < 1:
                phi2 = np.arcsin(rho2)
                cond = mixed_condition_second_order(gamma, m, gamma_m_ratio, phi2, sigma2_max=L_norm_approx) 
                if cond: 
                    phi_rho2_alpha_value = phi2
                else:
                    res.phi_rho2_alpha = alpha_prev
            else: 
                res.phi_rho2_alpha = alpha_prev
        res.phi_rho2.append(phi_rho2_alpha_value)

        # Skar condition
        cond = np.all(gamma*2 >= 2 * m * A.sum(axis=1))
        res.skar.append(cond)

        if res.J_alpha is not None:
            break

    res.skar_alpha = alphas[np.where(res.skar)[0][-1]] if any(res.skar) else None

    return res