"""
Solving 1d scalar wave equation with boundary data by PINNs

@date: 2022-03-18
@author: chazen
"""

import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional  as F
from matplotlib import pyplot as plt 

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DNN(nn.Module):
    """Fully connected neural network
    """
    def __init__(self, layer_sizes):
        super(DNN, self).__init__()
        self.layer_sizes = layer_sizes
        self.linears = nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))

    def forward(self, x):
        for linear in self.linears[:-1]:
            # test different activation function
            # x = F.sigmoid(linear(x))
            # x = F.relu(linear(x))
            x = F.softplus(linear(x))
            # x = F.tanh(linear(x))
        x = self.linears[-1](x)
        return x 


class PINN(nn.Module):
    """Physics imformed neural network
    """
    def __init__(self, X0, Xb, Ub, Xf, F, layer_sizes, C=1.):
        super(PINN, self).__init__()
        self.C = C 

        # Deal with initial condition(zero initial condition here)
        self.x0 = torch.tensor(X0[:, :1], requires_grad=True, 
                        dtype=torch.float32, device=device)
        self.t0 = torch.tensor(X0[:, 1:], requires_grad=True, 
                        dtype=torch.float32, device=device)

        # Deal with boundary condition, the solution at boundary 
        # is pre-computed by a certain numerical method with PML
        self.xb = torch.tensor(Xb[:, :1], requires_grad=True,
                        dtype=torch.float32, device=device)
        self.tb = torch.tensor(Xb[:, 1:], requires_grad=True, 
                        dtype=torch.float32, device=device)
        self.ub = torch.tensor(Ub, dtype=torch.float32, device=device)
        assert self.xb.dim == self.tb.dim == self.ub.dim == 2
        assert self.xb.shape[0] == self.tb.shape[0] == self.ub.shape[0]
        assert self.xb.shape[1] == self.tb.shape[1] == self.ub.shape[1] == 1

        # Collocation points
        self.xf = torch.tensor(Xf[:, :1], requires_grad=True,
                        dtype=torch.float32, device=device)
        self.tf = torch.tensor(Xf[:, 1:], requires_grad=True, 
                        dtype=torch.float32, device=device)
        self.f = torch.tensor(F, dtype=torch.float32, device=device)

        self.dnn = DNN(layer_sizes).to(device)

        # Optimizer, test different optimizer and learning rate
        self.optimizer = torch.optim.Adam(
            self.dnn.parameters,
            lr = 0.1
        )
        self.max_num_iters = 1000

    def net_u(self, x, t):
        u = self.dnn(torch.cat((x, t), dim=1))
        return u 

    def net_f(self, x, t, f):
        u = self.net(u, x, t)

        u_t = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_tt = torch.autograd.grad(
            u_t, t, 
            grad_outputs=torch.ones_like(u_t),
            retain_graph=True,
            create_graph=True
        )[0]

        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x, 
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        Lu = u_tt / self.C**2 - u_xx - f
        return Lu 

    def compute_loss(self):
        # Loss of collocation points
        Lu = self.net_f(self.xf, self.tf, self.f)
        lf = torch.mean(torch.square(Lu))

        # Loss of initial condition
        u0 = self.net_u(self.x0, self.t0)
        u0_t = torch.autograd.grad(
            u0, self.t0, 
            grad_outputs=torch.ones_like(u0),
            retain_graph=True,
            create_graph=True
        )[0]
        l0 = torch.mean(torch.square(u0)) + torch.mean(torch.square(u0_t))

        # Loss of boundary data
        ub = self.net_u(self.xb, self.ub)
        lb = torch.mean(torch.square(ub - self.ub))
        l = lb + l0 + lf 
        return l, lb, l0, lf

    def train(self, print_interval=10):
        self.dnn.train()
        loss = []
        loss0 = []
        lossb = []
        lossf = []
        for n_iter in range(self.max_num_iters):
            self.optimizer.zero_grad()

            l, lb, l0, lf = self.compute_loss()
            loss.append(l)
            loss0.append(l0)
            lossb.append(lb)
            lossf.append(lf)
            if (n_iter+1) % print_interval == 0:
                print('Iter %d, Loss %.5e, Lossb: %.5e, Loss0: %.5e, Lossf: %.5e'
                % (l.item(), lb.item(), l0.item(), lf.item()))

            l.backward()
            self.optimizer.step()


    def predict(self, x, t):
        u = self.net_u(x, t).detach().numpy()
        return u 


def gen_data(n0, nb, nf):
    # Initial data
    t0 = np.zeros((n0, 1))
    x0 = np.random.rand(n0, 1)
    X0 = np.concatenate((x0, t0), axis=1)

    # Collocation points
    Xb = 


    # Boundary data 


    
    return X0, Xb, Ub, Xf, F
        





