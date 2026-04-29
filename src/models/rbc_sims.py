import numpy as np
from src.base.model_base import DSGEModel

class RBCSimsModel(DSGEModel):
    def equations(self, x_m, x, x_p, e, params):
        # y: 0, c: 1, i: 2, l: 3, k: 4, w: 5, r: 6, z: 7, g: 8, b: 9
        y_m, c_m, i_m, l_m, k_m, w_m, r_m, z_m, g_m, b_m = x_m
        y, c, i, l, k, w, r, z, g, b = x
        y_p, c_p, i_p, l_p, k_p, w_p, r_p, z_p, g_p, b_p = x_p
        
        alpha = params['alpha']
        beta = params['beta']
        delta = params['delta']
        psi = params['psi']
        chi = params['chi']
        rho_z = params['rho_z']
        rho_g = params['rho_g']
        rho_b = params['rho_b']
        
        res = np.zeros(10)
        # 1. Euler
        res[0] = (1/c) * b - beta * (1/c_p) * b_p * (r_p + 1 - delta)
        # 2. Labor supply
        res[1] = psi * l**chi * c - w
        # 3. Wage
        res[2] = w - (1-alpha) * y / l
        # 4. Interest
        res[3] = r - alpha * y / k_m
        # 5. Resource
        res[4] = y - c - i - g
        # 6. Capital
        res[5] = k - (1-delta)*k_m - i
        # 7. Production
        res[6] = y - z * k_m**alpha * l**(1-alpha)
        # 8. Tech Shock
        res[7] = np.log(z) - rho_z * np.log(z_m) - e[0]
        # 9. Govt Shock
        res[8] = np.log(g/params['g_ss']) - rho_g * np.log(g_m/params['g_ss']) - e[1]
        # 10. Pref Shock
        res[9] = np.log(b) - rho_b * np.log(b_m) - e[2]
        
        return res

    def solve_ss(self):
        alpha = self.params['alpha']
        beta = self.params['beta']
        delta = self.params['delta']
        psi = self.params['psi']
        chi = self.params['chi']
        gy_ratio = self.params.get('gy_ratio', 0.2)
        
        z = 1.0
        b = 1.0
        r = 1/beta - 1 + delta
        ky = alpha / r
        iy = delta * ky
        cy = 1 - iy - gy_ratio
        
        # l^(1+chi) = (1-alpha) / (psi * cy)
        l = ((1-alpha) / (psi * cy))**(1/(1+chi))
        # y = (z * ky^alpha * l^(1-alpha))^(1/(1-alpha))
        y = (z * ky**alpha * l**(1-alpha))**(1/(1-alpha))
        k = ky * y
        c = cy * y
        i = iy * y
        w = (1-alpha) * y / l
        g = gy_ratio * y
        
        self.params['g_ss'] = g
        self.ss = np.array([y, c, i, l, k, w, r, z, g, b])
        return self.ss
