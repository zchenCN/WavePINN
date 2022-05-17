"""
Physics Informed Neural Networks for solving wave equations 
with two time snapshot wavefield data
@data: 2022-05-11
"""

import timeit
import torch
import torch.nn as nn
import numpy as np

import sys 
sys.path.append('..')
from gradient import GradientLayer

class PINN:
    """
    Build a physics informed neural network (PINN) model for the wave equation.
    Attributes:
        network: pytorch network model with input (t, x) and output u(t, x).
        c: wave velocity.
        grads: gradient layer.
    """

    def __init__(self, network, TX_init, U_init, TX_pde, TX_test, U_test, c=1.0):
        """
        Args:
            network: pytorch network model with input (t, x) and output u(t, x).
            c: wave velocity. Default is 1.
        """
        self.network = network
        self.c = c
        self.device = next(network.parameters()).device
        self.grads = GradientLayer(self.network)

        # Data preprocess
        self.uscl = np.abs(U_init).max() # normalize u in [-1, 1]

        assert TX_init.shape[0] == U_init.shape[0]
        assert TX_init.shape[1] == 2 and U_init.shape[1] == 1
        self.t_init = torch.tensor(TX_init[:, 0:1], dtype=torch.float64, requires_grad=True, device=self.device)
        self.x_init = torch.tensor(TX_init[:, 1:2], dtype=torch.float64, requires_grad=True, device=self.device)
        self.U_init = torch.tensor(U_init/self.uscl, dtype=torch.float64, device=self.device)

        # assert TX_bnd.shape[0] == U_bnd.shape[0]
        # assert TX_bnd.shape[1] == 2 and U_bnd.shape[1] == 1
        # self.t_bnd = torch.tensor(TX_bnd[:, 0:1], dtype=torch.float64, requires_grad=True, device=self.device)
        # self.x_bnd = torch.tensor(TX_bnd[:, 1:2], dtype=torch.float64, requires_grad=True, device=self.device)
        # self.U_bnd = torch.tensor(U_bnd/self.uscl, dtype=torch.float64, device=self.device)

        assert TX_pde.shape[1] == 2
        self.t_pde = torch.tensor(TX_pde[:, 0:1], dtype=torch.float64, requires_grad=True, device=self.device)
        self.x_pde = torch.tensor(TX_pde[:, 1:2], dtype=torch.float64, requires_grad=True, device=self.device)

        self.TX_test = TX_test 
        self.U_test = U_test

        # Optimizer
        self.num_epoch = 0
        self.optimizer_lbfgs = torch.optim.LBFGS(
            network.parameters(), 
            lr=1.0, 
            max_iter=15000,#15000, 
            max_eval=15000, 
            history_size=50,
            tolerance_grad=1e-5, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"       # can be "strong_wolfe"
            )

        self.optimizer_adam = torch.optim.Adam(
            network.parameters(), 
            lr=0.001
        )
        self.loss_list = list()
        self.loss_pde_list = list()
        self.loss_init_list = list()
        self.loss_bnd_list = list()
        self.error_list = list()

    def loss_pde(self):
        # pde loss
        _, _, _, d2u_dt2, d2u_dx2 = self.grads(self.t_pde, self.x_pde)
        eqn = d2u_dt2 - self.c*self.c * d2u_dx2
        loss_pde_val = torch.mean(torch.square(eqn))
        return loss_pde_val 

    def loss_init(self):
        # initial loss
        u_init, _, _, _, _ = self.grads(self.t_init, self.x_init)
        loss_init_val = torch.mean(torch.square(u_init - self.U_init))       
        return loss_init_val 

    def loss_bnd(self):
        # boundary loss
        u_bnd, _, _, _, _ = self.grads(self.t_bnd, self.x_bnd)
        loss_bnd_val = torch.mean(torch.square(u_bnd - self.U_bnd))
        return loss_bnd_val

    def loss(self, wp=0.00001, wi=1.0, wb=0.0):
        loss_pde_val = self.loss_pde()
        loss_init_val = self.loss_init()
        # loss_bnd_val = self.loss_bnd()
        loss_val = wp*loss_pde_val + wi*loss_init_val # + wb*loss_bnd_val

        self.optimizer_lbfgs.zero_grad()
        loss_val.backward()
        if self.num_epoch % 100 == 0:
            self.stop = timeit.default_timer()
            self.loss_list.append(loss_val.item())
            self.loss_init_list.append(loss_init_val.item())
            #self.loss_bnd_list.append(loss_bnd_val.item())
            self.loss_pde_list.append(loss_pde_val.item())
            error_val = self.error()
            self.error_list.append(error_val)
            print(f'Epoch: {self.num_epoch}'.ljust(12), f'Error: {error_val:.4f}',f'  Loss: {loss_val.item():.8f}', f'  Loss pde: {loss_pde_val.item():.8f}',
            f'  Loss init: {loss_init_val.item():.8f}', f'  Time: {self.stop-self.start:.1f}')
        self.num_epoch += 1
        return loss_val

    def train(self):
        self.network.train()
        self.start = timeit.default_timer()
        # while self.num_epoch < 2500:
        #     loss_init_val = self.loss_init()
        #     if self.num_epoch % 100 == 0:
        #         self.stop = timeit.default_timer()
        #         print(f'Epoch: {self.num_epoch}'.ljust(12), f'Loss: {loss_init_val.item():.8f}', f'Time: {self.stop-self.start:.1f}')
        #     self.optimizer_adam.zero_grad()
        #     loss_init_val.backward()
        #     self.optimizer_adam.step()
        #     self.num_epoch += 1
        self.optimizer_lbfgs.step(self.loss)

    def predict(self, TX):
        TX = torch.tensor(TX, dtype=torch.float64, device=self.device)
        U_pred = self.network(TX).detach().cpu().numpy()
        return U_pred * self.uscl

    def error(self):
        U_pred = self.predict(self.TX_test)
        error_val = np.linalg.norm(U_pred - self.U_test) / np.linalg.norm(self.U_test)
        return error_val