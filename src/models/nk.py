import numpy as np
from src.base.model_base import DSGEModel

class NKModel(DSGEModel):
    def equations(self, x_m, x, x_p, e, params):
        # y: 0, pi: 1, i: 2, g: 3, z: 4
        y_m, pi_m, i_m, g_m, z_m = x_m
        y, pi, i, g, z = x
        y_p, pi_p, i_p, g_p, z_p = x_p
        
        beta = params['beta']
        sigma = params['sigma']
        kappa = params['kappa']
        phi_pi = params['phi_pi']
        phi_y = params['phi_y']
        rho_g = params['rho_g']
        rho_z = params['rho_z']
        
        res = np.zeros(5)
        # 1. IS Curve: y = y_p - (1/sigma)*(i - pi_p) + g
        res[0] = y - y_p + (1/sigma)*(i - pi_p) - g
        # 2. Phillips Curve: pi = beta*pi_p + kappa*y + z
        res[1] = pi - beta*pi_p - kappa*y - z
        # 3. Taylor Rule: i = phi_pi*pi + phi_y*y
        res[2] = i - phi_pi*pi - phi_y*y
        # 4. Govt Shock: g = rho_g * g_m + e[0]
        res[3] = g - rho_g * g_m - e[0]
        # 5. Tech Shock: z = rho_z * z_m + e[1]
        res[4] = z - rho_z * z_m - e[1]
        
        return res

    def solve_ss(self):
        # Linear model, ss is 0
        self.ss = np.zeros(5)
        return self.ss
