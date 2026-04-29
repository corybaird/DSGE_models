import numpy as np
from src.base.model_base import DSGEModel

class RBCModel(DSGEModel):
    def equations(self, x_m, x, x_p, e, params):
        y_m, c_m, k_m, z_m = x_m
        y, c, k, z = x
        y_p, c_p, k_p, z_p = x_p
        
        alpha = params['alpha']
        beta = params['beta']
        delta = params['delta']
        rho = params['rho']
        
        res = np.zeros(4)
        res[0] = beta * (c/c_p) * (alpha * y_p / k + 1 - delta) - 1
        res[1] = y - z * k_m**alpha
        res[2] = k - (1-delta)*k_m - (y - c)
        res[3] = np.log(z) - rho * np.log(z_m) - e[0]
        
        return res

    def solve_ss(self):
        alpha = self.params['alpha']
        beta = self.params['beta']
        delta = self.params['delta']
        
        z = 1.0
        r = 1/beta - 1 + delta
        ky = alpha / r
        y = ky**(alpha/(1-alpha))
        k = ky * y
        c = y - delta * k
        
        self.ss = np.array([y, c, k, z])
        return self.ss
