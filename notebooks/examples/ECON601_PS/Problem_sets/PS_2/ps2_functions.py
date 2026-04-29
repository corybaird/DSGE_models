from textwrap import dedent
from scipy.stats import norm
import sympy as sy
from sympy.abc import a,c,b
import matplotlib.pyplot as plt
import numpy as np


class consum_funcs(object):
    def __init__(self, function):
        self.function = function
    
    def __str__(self):
        m = """\
       Function:{a}
          - ARA: -``u(c)/`u(c)
          - CRA: -``u(c)*c/`u(c)
        """
        return dedent(m.format(a=self.function))
    
    @property
    def compute_derivatives(self):
        u = self.function #Original function
        u1d = self.function.diff(c) #1st derivative
        u2d = self.function.diff(c,c) #2nd derivative
        
        return [u1d, u2d]
        

    def ara(self):
        ara_solu = -(self.compute_derivatives[1]/self.compute_derivatives[0])
        return ara_solu
    
    
    def rra(self):
        rra_solu = -(self.compute_derivatives[1]/self.compute_derivatives[0])*c
        return rra_solu
    
    def plot_rra_ara(self):
        ara_solu = -(self.compute_derivatives[1]/self.compute_derivatives[0])
        rra_solu = -(self.compute_derivatives[1]/self.compute_derivatives[0])*c
  
        ara_func, rra_func= [sy.lambdify((a,c),x) for x in [ara_solu, rra_solu]]