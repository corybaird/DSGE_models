import yaml
import numpy as np
import pydsge
from pathlib import Path
from src.models.nk import NKModel
from src.optimizer.solver import DSGESolver
from src.examples.pydsge_nk import NKPyDSGEModel

class ComparisonNK:
    def run(self):
        with open("params.yaml") as f:
            params = yaml.safe_load(f)['model']
        
        # 1. Native Solution
        model = NKModel(params)
        solver = DSGESolver(model)
        G_native, H_native, ss = solver.solve(m=2)
        irfs_native = solver.irfs(G_native, H_native, horizon=10)
        
        # 2. PyDSGE Solution
        nk_pydsge = NKPyDSGEModel()
        # pydsge.simulate() or .irfs()
        # Let's use the internal simulate to get IRFs
        # In pydsge, irfs() returns a dataframe
        irfs_pydsge = nk_pydsge.model.irfs(shocklist=('e_g',), horizon=10)
        
        print("Comparison of Inflation (pi) IRF to Demand Shock (g):")
        # Native: [horizon, n, m]. pi is index 1, g is index 0.
        native_pi_g = irfs_native[:, 1, 0]
        # pydsge returns a dict or dataframe. Let's check.
        # Usually it's a dict of dataframes.
        try:
            pydsge_pi_g = irfs_pydsge['e_g']['pi'].values
            print(f"Native: {native_pi_g[:5]}")
            print(f"PyDSGE: {pydsge_pi_g[:5]}")
            
            diff = np.abs(native_pi_g - pydsge_pi_g).max()
            print(f"\nMax absolute difference: {diff:.6e}")
            
            if diff < 1e-10:
                print("Native solver is CONSISTENT with PyDSGE for the NK model.")
            else:
                print("Discrepancy detected.")
        except Exception as e:
            print(f"Extraction failed: {e}")
            print(f"PyDSGE IRF result type: {type(irfs_pydsge)}")
            if hasattr(irfs_pydsge, 'columns'):
                print(f"Columns: {irfs_pydsge.columns}")

if __name__ == "__main__":
    comp = ComparisonNK()
    comp.run()
