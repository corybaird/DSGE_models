import yaml
import numpy as np
import pydsge
from pathlib import Path
from src.models.rbc import RBCModel
from src.optimizer.perturbation import PerturbationSolver

class ComparisonTest:
    def run(self):
        with open("params.yaml") as f:
            params = yaml.safe_load(f)['model']
        
        # 1. Native Solution
        model = RBCModel(params)
        ss = model.solve_ss()
        solver = PerturbationSolver()
        jac_m, jac_x, jac_p, jac_e = solver.get_jacobians(model.equations, ss, params, m=1)
        G_native, H_native = solver.solve_linear_system(jac_m, jac_x, jac_p, jac_e)
        
        # 2. PyDSGE Solution
        # We'll use the rbc.yaml we created earlier in data/temp/
        yaml_path = Path("data/temp/rbc.yaml")
        # Ensure it exists
        if not yaml_path.exists():
            print("pydsge yaml not found. Run the pipeline first.")
            return
            
        try:
            dsge = pydsge.DSGE.read(str(yaml_path))
            dsge.get_sys()
            # pydsge stores the solution in T and J
            # G matrix in pydsge is T
            G_pydsge = dsge.T
            
            print("Comparison of G matrix (first row):")
            print(f"Native: {G_native[0, :]}")
            print(f"PyDSGE: {G_pydsge[0, :]}")
            
            diff = np.abs(G_native - G_pydsge).max()
            print(f"Max absolute difference: {diff:.6e}")
            
        except Exception as e:
            print(f"PyDSGE failed: {e}")

if __name__ == "__main__":
    test = ComparisonTest()
    test.run()
