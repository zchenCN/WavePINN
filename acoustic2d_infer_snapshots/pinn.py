"""
Physics Informed Neural Networks for solving 2d acoustic 
wave equations with two time snapshot wavefield data

@data: 2022-05-17
"""

import timeit
import torch
import torch.nn as nn
import numpy as np
import tqdm

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

    def __init__(self, network, TX_init, U_init, TX_pde, TX_test, U_test, c=1.5):
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
        assert TX_init.shape[1] == 3 and U_init.shape[1] == 1
        self.t_init = torch.tensor(TX_init[:, 0:1], dtype=torch.float64, requires_grad=True, device=self.device)
        self.x_init = torch.tensor(TX_init[:, 1:2], dtype=torch.float64, requires_grad=True, device=self.device)
        self.z_init = torch.tensor(TX_init[:, 2:3], dtype=torch.float64, requires_grad=True, device=self.device)
        self.U_init = torch.tensor(U_init/self.uscl, dtype=torch.float64, device=self.device)

        assert TX_pde.shape[1] == 3
        self.t_pde = torch.tensor(TX_pde[:, 0:1], dtype=torch.float64, requires_grad=True, device=self.device)
        self.x_pde = torch.tensor(TX_pde[:, 1:2], dtype=torch.float64, requires_grad=True, device=self.device)
        self.z_pde = torch.tensor(TX_pde[:, 2:3], dtype=torch.float64, requires_grad=True, device=self.device)

        self.TX_test = TX_test 
        self.U_test = U_test

        # Optimizer
        self.max_num_epoch = 400000
        self.interval = 1000
        self.num_epoch = 0
        self.optimizer_lbfgs = torch.optim.LBFGS(
            network.parameters(), 
            lr=1.0, 
            max_iter=self.max_num_epoch,
            max_eval=50000, 
            history_size=50,
            tolerance_grad=1e-5, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"       # can be "strong_wolfe"
            )

        self.optimizer_adam = torch.optim.Adam(
            network.parameters(), 
            lr=0.0001
        )
        self.loss_list = list()
        self.loss_pde_list = list()
        self.loss_init_list = list()
        self.error_list = list()

    def loss_pde(self):
        # pde loss
        _, _, _, _, d2u_dt2, d2u_dx2, d2u_dz2 = self.grads(self.t_pde, self.x_pde, self.z_pde)
        eqn = d2u_dt2 - self.c*self.c * (d2u_dx2 + d2u_dz2)
        loss_pde_val = torch.mean(torch.square(eqn))
        return loss_pde_val 

    def loss_init(self):
        # initial loss
        # u_init, *_ = self.grads(self.t_init, self.x_init, self.z_init)
        TX_init = torch.cat((self.t_init, self.x_init, self.z_init), dim=1)
        u_init = self.network(TX_init)
        loss_init_val = torch.mean(torch.square(u_init - self.U_init))       
        return loss_init_val 

    def loss_lbfgs(self, wp=0.1, wi=1.0):
        loss_pde_val = self.loss_pde()
        loss_init_val = self.loss_init()
        loss_val = wp*loss_pde_val + wi*loss_init_val

        if self.num_epoch % self.interval == 0:
            self.stop = timeit.default_timer()
            self.loss_list.append(loss_val.item())
            self.loss_init_list.append(loss_init_val.item())
            self.loss_pde_list.append(loss_pde_val.item())
            error_val = self.error()
            self.error_list.append(error_val)
            print(f'Epoch: {self.num_epoch}'.ljust(12), f'   Error: {error_val:.4f}',f'  Loss: {loss_val.item():.8f}', f'  Loss pde: {loss_pde_val.item():.8f}',
            f'  Loss init: {loss_init_val.item():.8f}', f'  Time: {self.stop-self.start:.1f}')
        self.optimizer_lbfgs.zero_grad()
        loss_val.backward()
        self.num_epoch += 1
        return loss_val       

    def loss(self, wp=0.000001, wi=1.0): # [wp, error]: [1.0, ]
        loss_pde_val = self.loss_pde()
        loss_init_val = self.loss_init()
        loss_val = wp*loss_pde_val + wi*loss_init_val


        if self.num_epoch % self.interval == 0:
            self.stop = timeit.default_timer()
            self.loss_list.append(loss_val.item())
            self.loss_init_list.append(loss_init_val.item())
            self.loss_pde_list.append(loss_pde_val.item())
            error_val = self.error()
            self.error_list.append(error_val)
            # print(f'Epoch: {self.num_epoch}'.ljust(12), f'Error: {error_val:.4f}',f'  Loss: {loss_val.item():.8f}', f'  Loss pde: {loss_pde_val.item():.8f}',
            # f'  Loss init: {loss_init_val.item():.8f}', f'  Time: {self.stop-self.start:.1f}')
            print(f'Epoch: {self.num_epoch}'.ljust(12), f'Error: {error_val:.4f}',f'  Loss: {loss_val.item():.8f}', f'  Loss pde: {loss_pde_val.item():.8f}',
            f'  Loss init: {loss_init_val.item():.8f}', f'  Time: {self.stop-self.start:.1f}', file=self.file)
        return loss_val

    def train(self, mode='adam'):
        self.network.train()
        self.start = timeit.default_timer()
        output_path = './log.txt'
        self.file = open(output_path, 'w')

        if mode == 'lbfgs':
            self.optimizer_lbfgs.step(self.loss_lbfgs)
        else:
            pbar = tqdm.tqdm(range(self.max_num_epoch), unit='epoch')
            for self.num_epoch in pbar:
            # for self.num_epoch in range(self.max_num_epoch):
                self.network.train()
                loss_val = self.loss()
                self.optimizer_adam.zero_grad()
                loss_val.backward()
                self.optimizer_adam.step()
                pbar.set_description(f'Loss = {loss_val.item():.8f}')
        self.file.close()

    def predict(self, TX):
        # assert np.allclose(TX, np.zeros_like(TX)) == False
        TX = torch.tensor(TX, dtype=torch.float64, device=self.device)
        U_pred = self.network(TX).detach().cpu().numpy()
        return U_pred * self.uscl

    def error(self):
        U_pred = self.predict(self.TX_test)
        assert U_pred.shape == self.U_test.shape
        error_val = np.linalg.norm(U_pred - self.U_test) / np.linalg.norm(self.U_test)
        return error_val
