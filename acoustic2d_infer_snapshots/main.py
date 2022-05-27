import numpy as np
from scipy.interpolate import griddata
import torch 
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from pinn import PINN
from torchsummary import summary

import sys 
import os
sys.path.append('..')
from fnn import FNN 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## ---------------------- 1. Data Preprocess---------------------------------------------------
# ------------------ Homogenenous model ----------------------------------
# h = 0.01 # km
# dt = 0.001 # s
# nx = nz = 200 
# xmax = h * nx 
# zmax = h * nz

# nt1 = 200
# nt2 = 300
# nt3 = 500
# nt4 = 850
# t1 = 0.0
# t2 = (nt2 - nt1) * dt 
# t3 = (nt3 - nt1) * dt 
# t4 = (nt4 - nt1) * dt 

# # Load data 
# u = list()
# for i in range(4):
#     data = np.loadtxt('./wavefield2d_t' + str(i) + '.txt')
#     u.append(data)

# # Grid
# xx, zz = np.meshgrid(np.arange(nx+1)*h, np.arange(nz+1)*h)
# xx = xx.flatten()[:, np.newaxis]
# zz = zz.flatten()[:, np.newaxis]
# X = np.c_[xx, zz]
# --------------------- Homogenenous model ---------------------------------------

# ------------------------- Marmousi model ---------------------------------------
h = 0.01 # km
dt = 0.001 # s
nx = nz = 200 
xmax = h * nx 
zmax = h * nz

nt1 = 200
nt2 = 250
nt3 = 350
nt4 = 450
t1 = 0.0
t2 = (nt2 - nt1) * dt 
t3 = (nt3 - nt1) * dt 
t4 = (nt4 - nt1) * dt 

# Load data 
u = list()
for i in range(4):
    data = np.loadtxt('./marmousi_t' + str(i) + '.txt')
    u.append(data)

marmousi = np.loadtxt('marmousi.txt')

# Grid
xx, zz = np.meshgrid(np.arange(nx+1)*h, np.arange(nz+1)*h)
xx = xx.flatten()[:, np.newaxis]
zz = zz.flatten()[:, np.newaxis]
X = np.c_[xx, zz]
# ------------------------- Marmousi model ---------------------------------------

# Initial data
num_init = 40 # number of initial data per time snapshot is n_init^2
x_init, z_init = np.meshgrid(np.linspace(0, xmax, num_init), np.linspace(0, zmax, num_init))
X_init = np.c_[x_init.flatten()[:, np.newaxis], z_init.flatten()[:, np.newaxis]] 
# t1 
t_init1 = t1 * np.ones((num_init**2, 1))
TX_init1 = np.c_[t_init1, X_init]
U_init1 = griddata(X, u[0].flatten(), X_init)[:, np.newaxis] # interpolate wavefield
# t2
t_init2 = t2 * np.ones((num_init**2, 1))
TX_init2 = np.c_[t_init2, X_init]
U_init2 = griddata(X, u[1].flatten(), X_init)[:, np.newaxis] # interpolate wavefield

TX_init = np.r_[TX_init1, TX_init2]
U_init = np.r_[U_init1, U_init2]

# Test data
# t3 
t_init3 = t3 * np.ones((num_init**2, 1))
TX_init3 = np.c_[t_init3, X_init]
U_init3 = griddata(X, u[2].flatten(), X_init)[:, np.newaxis] # interpolate wavefield
# t4
t_init4 = t4 * np.ones((num_init**2, 1))
TX_init4 = np.c_[t_init4, X_init]
U_init4 = griddata(X, u[3].flatten(), X_init)[:, np.newaxis] # interpolate wavefield

TX_test = np.r_[TX_init3, TX_init4]
U_test = np.r_[U_init3, U_init4]


# Collocation points
num_pde = 5000
TX_pde = np.random.rand(num_pde, 3)
TX_pde[:, 0:1] = TX_pde[:, 0:1] * 1.2*t4 
TX_pde[:, 1:3] = TX_pde[:, 1:3] * 2 # x and z in [0, 2], normalization later??

X_pde = TX_pde[:, 1:3]
c = griddata(X, marmousi.flatten()[:, np.newaxis], X_pde)
assert len(c) == len(X_pde)

# ----------------------------- 2. Model and Train -------------------------------------------------

layer_sizes = [3] + [32, 16, 16, 32] + [1] # error = 0.15
# layer_sizes = [3] + [64] * 5 + [1]
model = FNN(layer_sizes)
os.system('clear')
summary(model, (3, ), device='cpu')
model.to(device).to(torch.float64)

pinn = PINN(model, TX_init, U_init, TX_pde, TX_test, U_test, c=c)
pinn.train()
torch.save(pinn.network, 'model.pt')
# ----------------------------- 3. Plotting -------------------------------------

vmin, vmax = -pinn.uscl, pinn.uscl

# plot real data
fig1 = plt.figure(figsize=(14,8))
gs = GridSpec(2, 4)
plt.subplot(gs[0, 0])
plt.imshow(u[0], cmap='rainbow')
#plt.imshow(U_init1.reshape(num_init, num_init), cmap='rainbow')
#plt.pcolormesh(x_init, z_init, U_init1.reshape(num_init, num_init), cmap='rainbow')
plt.xticks([])
plt.yticks([])
plt.subplot(gs[0, 1])
plt.imshow(u[1], cmap='rainbow')
#plt.imshow(U_init2.reshape(num_init, num_init), cmap='rainbow')
#plt.pcolormesh(x_init, z_init, U_init2.reshape(num_init, num_init), cmap='rainbow')
plt.xticks([])
plt.yticks([])
plt.subplot(gs[0, 2])
plt.imshow(u[2], cmap='rainbow')
# plt.imshow(U_init3.reshape(num_init, num_init), cmap='rainbow')
#plt.pcolormesh(x_init, z_init, U_init3.reshape(num_init, num_init), cmap='rainbow')
plt.xticks([])
plt.yticks([])
plt.subplot(gs[0, 3])
plt.imshow(u[3], cmap='rainbow')
# plt.imshow(U_init4.reshape(num_init, num_init), cmap='rainbow')
#plt.pcolormesh(x_init, z_init, U_init4.reshape(num_init, num_init), cmap='rainbow')
plt.xticks([])
plt.yticks([])

# plot predict data
t_list = [t1, t2, t3, t4]
for i in range(4):
    t_plot = np.ones((len(X), 1)) * t_list[i]
    TX_plot = np.c_[t_plot, X]
    u_plot = pinn.predict(TX_plot).reshape(nz+1, nx+1)
    plt.subplot(gs[1, i])
    plt.imshow(u_plot, cmap='rainbow')
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.savefig('result_img.png', transparent=True)

fig2, ax = plt.subplots(2, 2, figsize=(14, 8))
x_plot = np.arange(len(pinn.loss_list))* pinn.interval
ax[0, 0].plot(x_plot, pinn.loss_list)
ax[0, 0].set_title('Loss')
ax[0, 0].set_xlabel('Epoch')
ax[0, 0].set_yscale('log')
ax[0, 1].plot(x_plot, pinn.loss_pde_list)
ax[0, 1].set_title('PDE Loss')
ax[0, 1].set_yscale('log')
ax[0, 1].set_xlabel('Epoch')
ax[1, 0].plot(x_plot, pinn.loss_init_list)
ax[1, 0].set_title('Initial Loss')
ax[1, 0].set_xlabel('Epoch')
ax[1, 0].set_yscale('log')
ax[1, 1].plot(x_plot, pinn.error_list)
ax[1, 1].set_title('Error')
ax[1, 1].set_xlabel('Epoch')
ax[1, 1].set_yscale('log')
plt.tight_layout()
plt.savefig('loss_error.png', transparent=True)
plt.show()

