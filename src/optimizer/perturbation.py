import numpy as np
from scipy.linalg import qz, ordqz

class PerturbationSolver:
    def get_jacobians(self, func, ss, params, m=1):
        eps = 1e-8
        n = len(ss)
        e0 = np.zeros(m)
        
        f0 = func(ss, ss, ss, e0, params)
        
        jac_m = np.zeros((n, n))
        jac_x = np.zeros((n, n))
        jac_p = np.zeros((n, n))
        jac_e = np.zeros((n, m))
        
        for i in range(n):
            ss_m = ss.copy()
            ss_m[i] += eps
            jac_m[:, i] = (func(ss_m, ss, ss, e0, params) - f0) / eps
            
            ss_x = ss.copy()
            ss_x[i] += eps
            jac_x[:, i] = (func(ss, ss_x, ss, e0, params) - f0) / eps
            
            ss_p = ss.copy()
            ss_p[i] += eps
            jac_p[:, i] = (func(ss, ss, ss_p, e0, params) - f0) / eps
            
        for i in range(m):
            e_eps = e0.copy()
            e_eps[i] += eps
            jac_e[:, i] = (func(ss, ss, ss, e_eps, params) - f0) / eps
            
        return jac_m, jac_x, jac_p, jac_e

    def solve_linear_system(self, jac_m, jac_x, jac_p, jac_e):
        n = jac_m.shape[0]
        A = np.block([[jac_p, np.zeros((n, n))], [np.zeros((n, n)), np.eye(n)]])
        B = np.block([[-jac_x, -jac_m], [np.eye(n), np.zeros((n, n))]])
        
        S, T, alpha, beta, Q, Z = ordqz(A, B, sort='ouc')
        
        try:
            G = Z[:n, :n] @ np.linalg.inv(Z[n:, :n])
        except np.linalg.LinAlgError:
            return None, None
            
        # H = -(jac_p @ G + jac_x)^-1 @ jac_e
        try:
            H = -np.linalg.inv(jac_p @ G + jac_x) @ jac_e
        except np.linalg.LinAlgError:
            H = None
            
        return G, H

    def compute_irfs(self, G, H, horizon=20):
        n = G.shape[0]
        m = H.shape[1]
        irfs = np.zeros((horizon, n, m))
        
        # Shock at t=0
        for i in range(m):
            e = np.zeros(m)
            e[i] = 1.0
            x = H @ e
            irfs[0, :, i] = x
            for t in range(1, horizon):
                x = G @ x
                irfs[t, :, i] = x
                
        return irfs
