import numpy as np
import torch 
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from pinn import PINN
from torchsummary import summary

import sys 
sys.path.append('..')
from fnn import FNN 
from gradient import GradientLayer 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## ---------------------- 1. Data Preprocess---------------------------------------------------
nx = 1000
dx = 0.001 # km 
dt = 0.0001 # s 
nt1 = 2000
nt2 = 2400
nt3 = 4500
t1 = 0.0
t2 = (nt2 - nt1) * dt 
t3 = (nt3 - nt1) * dt 
xx = dx * np.arange(nx+1).reshape(-1, 1)
x_init = np.r_[xx, xx]

# Initial data
tt1 = t1 * np.ones((nx+1, 1), dtype=np.float64)
tt2 = t2 * np.ones((nx+1, 1), dtype=np.float64)
t_init = np.r_[tt1, tt2]
TX_init = np.c_[t_init, x_init]
u_init1 = np.loadtxt('./wavefields_t1.txt').reshape(-1, 1)
u_init2 = np.loadtxt('./wavefields_t2.txt').reshape(-1, 1)
U_init = np.r_[u_init1, u_init2]

# Test data
tt3 = t3 * np.ones((nx+1, 1), dtype=np.float64)
TX_test = np.c_[tt3, xx]
U_test = np.loadtxt('./wavefields_t3.txt').reshape(-1, 1)

# # Boundary data
# num_bnd = 2000
# TX_bnd = np.random.rand(num_bnd, 2)
# TX_bnd[:, 1:2] = np.round(TX_bnd[:, 1:2]) # x = 0 or 1
# U_bnd = np.zeros((num_bnd, 1), dtype=np.float64)

# Collocation points
num_pde = 1000
TX_pde = np.random.rand(num_pde, 2)

# num_pde = 10000
# t_pde = np.linspace(0, 1, num_pde, dtype=np.float64).reshape(-1, 1)
# x_pde = np.linspace(0, 1, num_pde, dtype=np.float64).reshape(-1, 1)
# TX_pde = np.c_[t_pde, x_pde]



# ----------------------------- 2. Model and Train -------------------------------------------------

layer_sizes = [2] + [32, 16, 16, 32] + [1]
model = FNN(layer_sizes)
summary(model, (2, ), device='cpu')
model.to(device).to(torch.float64)

pinn = PINN(model, TX_init, U_init, TX_pde, TX_test, U_test)
pinn.train()

# ----------------------------- 3. Plotting -------------------------------------



# predict u(t,x) distribution
num_test_samples = 1000
t_flat = np.linspace(0, 1, num_test_samples)
x_flat = np.linspace(0, 1, num_test_samples)
t, x = np.meshgrid(t_flat, x_flat)
tx = np.stack([t.flatten(), x.flatten()], axis=-1)
u = pinn.predict(tx)
u = u.reshape(t.shape)

# plot u(t,x) distribution as a color-map
fig = plt.figure(figsize=(14,8))
gs = GridSpec(2, 3)
plt.subplot(gs[0, :])
vmin, vmax = -pinn.uscl, pinn.uscl
plt.pcolormesh(t, x, u, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
plt.xlabel('t')
plt.ylabel('x')
cbar = plt.colorbar(pad=0.05, aspect=10)
cbar.set_label('u(t,x)')
cbar.mappable.set_clim(vmin, vmax)

# plot u(t=const, x) cross-sections

plt.subplot(gs[1, 0])
tx = np.c_[t1 * np.ones((nx+1, 1)), xx]
u = pinn.predict(tx)
plt.plot(xx.flatten(), u_init1, 'r--', label='init')
plt.plot(xx.flatten(), u, label='predict')
plt.legend()
plt.title('t={}'.format(t1))
plt.xlabel('x')
plt.ylabel('u(t,x)')

plt.subplot(gs[1, 1])
tx = np.c_[t2 * np.ones((nx+1, 1)), xx]
u = pinn.predict(tx)
plt.plot(xx.flatten(), u_init2, 'r--', label='init')
plt.plot(xx.flatten(), u, label='predict')
plt.legend()
plt.title('t={}'.format(t2))
plt.xlabel('x')
plt.ylabel('u(t,x)')

plt.subplot(gs[1, 2])
tx = np.c_[t3 * np.ones((nx+1, 1)), xx]
u = pinn.predict(tx)
plt.plot(xx.flatten(), U_test, 'r--', label='test')
plt.plot(xx.flatten(), u, label='predict')
plt.legend()
plt.title('t={}'.format(t3))
plt.xlabel('x')
plt.ylabel('u(t,x)')

plt.tight_layout()
plt.savefig('result_img_dirichlet.png', transparent=True)
#plt.show()

fig2, ax = plt.subplots(1, 2, figsize=(7, 4))
ax[0].plot(pinn.loss_list, label='loss')
ax[0].plot(pinn.loss_init_list, label='init')
# ax[0].plot(pinn.loss_pde_list, label='pde')
ax[0].plot(pinn.loss_bnd_list, label='bnd')
ax[0].legend()
ax[0].set_title('Loss')
ax[0].set_xlabel('Epoch/100')
ax[1].plot(pinn.error_list)
ax[1].set_title('Error')
ax[1].set_xlabel('Epoch/100')
plt.tight_layout()
plt.savefig('loss_error.png', transparent=True)
plt.show()

# change t1 data
# weighting wb=1.0 0.1 0.01 0.001
# num_pde
# Multiscale

# normalize t
