import yaml
from pathlib import Path
from src.models.rbc import RBCModel
from src.models.nk import NKModel
from src.optimizer.solver import DSGESolver
from src.models.rbc_sims import RBCSimsModel

class PipelineRunner:
    def run(self):
        config_path = Path("params.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)['model']
        
        # RBC Solve
        rbc = RBCModel(config)
        solver_rbc = DSGESolver(rbc)
        G_rbc, H_rbc, ss_rbc = solver_rbc.solve(m=1)
        
        # NK Solve
        nk = NKModel(config)
        solver_nk = DSGESolver(nk)
        G_nk, H_nk, ss_nk = solver_nk.solve(m=2)

        # Sims RBC Solve
        sims_rbc = RBCSimsModel(config)
        solver_sims = DSGESolver(sims_rbc)
        G_sims, H_sims, ss_sims = solver_sims.solve(m=3)
        
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        with open(reports_dir / "solve_summary.txt", "w") as f:
            f.write("DSGE Native Solver Summary\n")
            f.write("==========================\n\n")
            if G_rbc is not None:
                f.write(f"Standard RBC Solved. G[0,0]: {G_rbc[0,0]:.4f}\n")
            if G_nk is not None:
                f.write(f"NK Model Solved. G[0,0]: {G_nk[0,0]:.4f}\n")
            if G_sims is not None:
                f.write(f"Sims RBC Solved. G[0,0]: {G_sims[0,0]:.4f}\n")

if __name__ == "__main__":
    runner = PipelineRunner()
    runner.run()
