import sympy as sy
from sympy.abc import a,c,b
import matplotlib.pyplot as plt
import numpy as np


def consum_function_plot(function):
    u = function #Original function
    u1d = function.diff(c) #1st derivative
    u2d = function.diff(c,c) #2nd derivative
    u_func, u1d_func, u2d_func = [sy.lambdify((a,c),x) for x in [u, u1d, u2d]] #Convert to lambda functions
    
    #Numerically solve functions
    x = np.linspace(0,10,101)
    solu_all = []
    for func in [u_func, u1d_func, u2d_func]:
        solu_single = [] #Creates nested list
        for i in x:
            solu_single.append(func(1,i)) #appends to nested list
        solu_all.append(solu_single) #appends nested list to all list
    
    #Graph
    plt.plot(x,solu_all[0], label='U(c)')
    plt.plot(x,solu_all[1], label=r'$U\prime$')
    plt.plot(x, solu_all[2], label=r'$U\prime\prime$')
    plt.ylabel('U(c)')
    plt.xlabel('c')
    plt.axhline(u_func(1,0), color='red', label=r"$\frac{-1}{a}$")
    plt.legend();
    

def consum_function_solver(function, time=np.array([1,2])):
    #Functions
    u = function #Original function
    u1d = function.diff(c) #1st derivative
    u2d = function.diff(c,c) #2nd derivative
    u_func, u1d_func, u2d_func = [sy.lambdify((a,c),x) for x in [u, u1d, u2d]] #Convert to lambda functions
    
    #Numerically solve functions
    x = time
    solu_all = []
    for func in [u_func, u1d_func, u2d_func]:
        solu_single = [] #Creates nested list
        for i in x:
            solu_single.append(func(1,i)) #appends to nested list
        solu_all.append(solu_single) #appends nested list to all list
    
    return solu_all


def power_euler_nograph(beta=False, r=False, sigma=False):
     
    beta_r = beta*(1+r)
    euler_rhs = beta_r**sigma
    
    y1, y2 = [10,10]
    h = y1+(y2/(1+r))
    c1 = h/(1+(beta**sigma*(1+r)**(sigma-1)))
    c2 = (beta*(1+r)**sigma)*(c1)
    return list([c1, c2, euler_rhs])
    










