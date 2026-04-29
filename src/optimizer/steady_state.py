import numpy as np
from scipy.optimize import root

class SteadyStateSolver:
    def solve(self, model_func, initial_guess, params):
        res = root(model_func, initial_guess, args=(params,))
        return res.x if res.success else None
