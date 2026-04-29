import yaml
import numpy as np
from src.models.nk import NKModel
from src.optimizer.perturbation import PerturbationSolver

class CustomNKSolverTest:
    def run(self):
        with open("params.yaml") as f:
            params = yaml.safe_load(f)['model']
        
        model = NKModel(params)
        ss = model.solve_ss()
        print(f"Steady State: {ss}")
        
        solver = PerturbationSolver()
        # NK model has 2 shocks: g and z
        jac_m, jac_x, jac_p, jac_e = solver.get_jacobians(model.equations, ss, params, m=2)
        
        print("Jacobians computed successfully.")
        
        G, H = solver.solve_linear_system(jac_m, jac_x, jac_p, jac_e)
        if G is not None and H is not None:
            print("Policy functions G and H computed.")
            irfs = solver.compute_irfs(G, H, horizon=10)
            
            # Shocks: 0: g (demand), 1: z (supply/tech)
            # Variables: 0: y, 1: pi, 2: i, 3: g, 4: z
            print("IRF for Inflation (pi) to Demand Shock (g):")
            print(irfs[:, 1, 0])
            
            print("IRF for Output Gap (y) to Demand Shock (g):")
            print(irfs[:, 0, 0])
        else:
            print("Failed to compute policy functions.")

if __name__ == "__main__":
    tester = CustomNKSolverTest()
    tester.run()
