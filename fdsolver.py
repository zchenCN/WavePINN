"""
@brief: FDTD (Finite Difference Time Domain) method for wave equations
@data: 2022-04-14
@author: zchen
"""

import numpy as np
from matplotlib import pyplot as plt

def ricker(t, f0):
    """Ricker wavelet"""
    sigma = 1 / (np.pi * f0 * np.sqrt(2))
    t0 = 6 * sigma
    tmp = np.pi**2 * f0**2 * (t-t0)**2 
    w = (1 - 2*tmp) * np.exp(-tmp)
    return w

class FDTD1D:
    def __init__(self, h, dt, c, nt, x0, f0):
        self.nx = len(c) - 1
        self.h = h 
        self.dt = dt 
        self.nt = nt
        self.x0 = x0
        self.f0 = f0
        self.c = c
        self.cdt = c * dt 
        
        self.u_next = np.zeros(self.nx + 1)
        self.u_current = np.zeros(self.nx + 1)
        self.u_prev = np.zeros(self.nx + 1)
        self.wavefield = np.zeros((self.nt + 1, self.nx + 1))
        
#     def ricker(self, t):
#         sigma = 1 / (np.pi * self.f0 * np.sqrt(2))
#         t0 = 6 * sigma
#         tmp = np.pi**2 * self.f0**2 * (t-t0)**2 
#         w = (1 - 2*tmp) * np.exp(-tmp)
#         return w
    
    def cfl_number(self):
        cfl = self.c.max() * self.dt / self.h
        num_grid_per_wavelength = int(self.c.min() / (self.f0 * self.h))
        print(f'CFL number is {cfl}, number of grid points in each wavelength is about {num_grid_per_wavelength}')
        
    def abc(self):
        # Left boundary
        self.u_next[0] = self.cdt[0] * (self.u_current[1] - \
                        self.u_current[0]) / self.h + self.u_current[0]
        # Right boundary
        self.u_next[-1] = -self.cdt[-1] * (self.u_current[-1] - \
                        self.u_current[-2]) / self.h + self.u_current[-1]
        
    def deriv_xx(self, u):
        duxx = np.zeros_like(u)
        for i in range(1, self.nx):
            duxx[i] = (u[i+1] - 2 * u[i] + u[i-1]) / self.h ** 2
        return duxx
        
    def steps(self):
        for n in range(self.nt+1):
            self.wavefield[n] = self.u_next
            t = n * self.dt
            self.u_next = 2 * self.u_current - self.u_prev + self.cdt**2 * self.deriv_xx(self.u_current)
            # u_next[x0] += src(t)
            self.u_next[self.x0] += self.dt**2 * ricker(t, self.f0) #* 1e4 # Be careful!!!
            self.abc()
            self.u_current, self.u_prev = self.u_next, self.u_current

    def plot_wavefield_nt(self, nt):
        u_nt = self.wavefield[nt]
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(u_nt)
        ax.set_title(f'snapshot at {nt*self.dt:.2f}s ')
        plt.show()
        
    def plot_wavefield(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        im = ax.imshow(self.wavefield, cmap="gray", aspect="auto")
        ax.set_xlabel(r'n_x')
        ax.set_ylabel(r'n_t')
        plt.colorbar(im)
        plt.show()
        
