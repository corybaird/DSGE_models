import numpy as np
from src.optimizer.perturbation import PerturbationSolver

class DSGESolver:
    def __init__(self, model):
        self.model = model
        self.pert = PerturbationSolver()
        
    def solve(self, m=1):
        ss = self.model.solve_ss()
        params = self.model.params
        jac_m, jac_x, jac_p, jac_e = self.pert.get_jacobians(self.model.equations, ss, params, m=m)
        G, H = self.pert.solve_linear_system(jac_m, jac_x, jac_p, jac_e)
        return G, H, ss

    def irfs(self, G, H, horizon=20):
        return self.pert.compute_irfs(G, H, horizon=horizon)
