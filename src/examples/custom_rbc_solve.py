import yaml
import numpy as np
from src.models.rbc import RBCModel
from src.optimizer.perturbation import PerturbationSolver

class CustomSolverTest:
    def run(self):
        with open("params.yaml") as f:
            params = yaml.safe_load(f)['model']
        
        model = RBCModel(params)
        ss = model.solve_ss()
        print(f"Steady State: {ss}")
        
        solver = PerturbationSolver()
        jac_m, jac_x, jac_p, jac_e = solver.get_jacobians(model.equations, ss, params)
        
        print("Jacobians computed successfully.")
        
        G, H = solver.solve_linear_system(jac_m, jac_x, jac_p, jac_e)
        if G is not None and H is not None:
            print("Policy functions G and H computed.")
            irfs = solver.compute_irfs(G, H, horizon=10)
            print("IRF for Consumption (c) to Technology Shock (e_z):")
            print(irfs[:, 1, 0])
        else:
            print("Failed to compute policy functions.")

if __name__ == "__main__":
    tester = CustomSolverTest()
    tester.run()
