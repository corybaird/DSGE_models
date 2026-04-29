import yaml
import numpy as np
from src.models.rbc_sims import RBCSimsModel
from src.optimizer.solver import DSGESolver

class SimsRBCSolveTest:
    def run(self):
        with open("params.yaml") as f:
            params = yaml.safe_load(f)['model']
        
        model = RBCSimsModel(params)
        solver = DSGESolver(model)
        
        # 3 shocks: z, g, b
        G, H, ss = solver.solve(m=3)
        
        if G is not None:
            print("Sims RBC Model Solved Successfully.")
            print(f"Steady State (y, c, i, l, k, w, r, z, g, b):\n{ss}")
            
            irfs = solver.irfs(G, H, horizon=10)
            
            # Variables: 1: c, 3: l, 7: z (Shock 0), 8: g (Shock 1), 9: b (Shock 2)
            print("\nIRF for Consumption (c) to Tech Shock (z):")
            print(irfs[:, 1, 0])
            
            print("\nIRF for Consumption (c) to Govt Shock (g):")
            print(irfs[:, 1, 1])
            
            print("\nIRF for Consumption (c) to Pref Shock (b):")
            print(irfs[:, 1, 2])
        else:
            print("Failed to solve Sims RBC model.")

if __name__ == "__main__":
    tester = SimsRBCSolveTest()
    tester.run()
