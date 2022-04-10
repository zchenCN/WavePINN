"""
Finite difference solver for 1d scalar wave equation
Reference: https://pysit.readthedocs.io/en/latest/exercises/part_1.html

(1/c^2)u_tt - u_xx = f(x, t)

@data: 2021-12-07
@author: chazen
"""

import numpy as np
from matplotlib import pyplot as plt


def delta(x, x0):
    """Numerical Dirac delta function at x0
    """
    beta = 0.01
    exp = np.exp(-(x-x0)**2 / beta)
    return exp / np.sqrt(np.pi * beta)
    
def ricker(t, f0):
    """Ricker wavelet
    """
    sigma = 1 / (np.pi * f0 * np.sqrt(2))
    t0 = 6 * sigma
    tmp = np.pi**2 * f0**2 * (t-t0)**2 
    w = (1 - 2*tmp) * np.exp(-tmp)
    return w #* 1e2

def compute_f(x, t, x0, f0):
    """RHS(Nonhomogenenous term) of wave equaion 
    f(x, t) = w(t)*delta(x-x0)
    """
    d = delta(x, x0).reshape(1, -1)
    w = ricker(t, f0).reshape(-1, 1)
    f = np.dot(w, d)
    return f

# def point_source(source_time, x0, x):
#     """Point source at location x0, f(t, x) = w(t) * delta(x-x0)
#     """
#     st = source_time.reshape(-1, 1)
#     d = delta(x, x0).reshape(1, -1)
#     f = np.dot(st, d)
#     return f

class ABCSolver:
    """Solver for the 1D scalar acoustic wave equation with 
    absorbing boundary conditions(ABC)
    """
    def __init__(self, nx, dx, nt, dt, C, f):
        # Check CFL condition
        alpha = 1/6 # CFL constant
        cfl = dt * C.max() / dx
        if cfl > alpha:
            raise ValueError("The CFL number is too large, try to adjust dt and dx")
        self.nx = nx
        self.dx = dx
        self.nt = nt
        self.dt = dt 
        self.C = C
        self.f = f

    def construc_matrics(self):
        # Stifness matrix
        K = np.diag(np.ones(self.nx) * (2 / self.dx**2)) + \
            np.diag(np.ones(self.nx-1) * (-1 / self.dx**2), 1) + \
            np.diag(np.ones(self.nx-1) * (-1 / self.dx**2), -1)
        K[0, 0] = 1 / self.dx 
        K[0, 1] = -1 / self.dx
        K[-1, -1] = 1 / self.dx
        K[-1, -2] = -1 / self.dx
        
        # Attenuation matrix
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1 / self.C[0]
        A[-1, -1] = 1 / self.C[-1]
        
        # Mass matrix
        M = np.diag(1/ self.C**2)
        M[0, 0] = 0.0
        M[-1, -1] = 0.0
        
        return K, A, M
    
    def leaf_frog(self):
        # Source function
        # ts = np.arange(self.nt) * self.dt
        # xs = np.arange(self.nx) * self.dx
        # source_time = ricker(ts, f0)
        # source = point_source(source_time, x0, xs)

        K, A, M = self.construc_matrics()
        u = np.zeros((self.nt, self.nx)) # u(t, x)
        L = M / self.dt**2 + A / self.dt
        for n in range(self.nt-1):
            if n == 0:
                # f = source[0:1, :].T
                f = self.f[0:1, :].T
            else:
                unow = u[n:n+1, :].T
                uold = u[n-1:n, :].T
                f = self.f[n:n+1].T \
                    + (2 * M / self.dt**2 + A / self.dt - K) @ unow \
                    - (M / self.dt**2) @ uold
            unew = np.linalg.solve(L, f)
            u[n+1] = unew.squeeze()
        return u

def gen_data(C):
    f0 = 3.
    x0 = 0.5
    nx = 201
    dx = 1 / 200
    T = 1
    dt = (1/6) * dx / C.max()
    nt = int(T / dt)
    ts = np.arange(nt) * dt
    xs = np.arange(nx) * dx
    f = compute_f(xs, ts, x0, f0)
    solver = ABCSolver(nx, dx, nt, dt, C, f)
    u = solver.leaf_frog()
    fig = plt.imshow(u, cmap="gray", aspect="auto")
    plt.xticks(np.linspace(0, nx-1, 5), np.linspace(0, 200, 5)*dx)
    plt.yticks(np.linspace(0, nt-1, 5), np.linspace(0, nt-1, 5)*dt)
    plt.savefig("figures/1dwave.png")
    X, T = np.meshgrid(xs, ts)
    data = np.hstack((X.flatten()[:, np.newaxis], T.flatten()[:, np.newaxis], u.flatten()[:, np.newaxis]))
    np.savetxt('data/1dwave.txt', data)
    print('ns = %d, nt= %d, max absolute of u=%.4e, max absolute of f=%.4e' % (len(xs), len(ts), np.abs(u).max(), np.abs(f).max()))
    return u

if __name__ == "__main__":
    C0 = np.ones(201)
    u = gen_data(C0)
   
    #dC = -100.0*(xs-0.5)*np.exp(-((xs-0.5)**2)/(1e-4))
    #dC[np.where(abs(dC) < 1e-7)] = 0
    #C = C0 + dC





    
