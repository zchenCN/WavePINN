"""
Solving 1d scalar wave equation with ABC by PINNs

@date: 2021-12-17
@author: chazen
"""

import numpy as np 
import torch 
import torch.nn as nn
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
            x = torch.tanh(linear(x))
        x = self.linears[-1](x)
        return x 


class PINN(nn.Module):
    """Physic informed neural network
    """
    def __init__(self, X_i, T, X_f, F, layer_sizes, C=1.):
        super(PINN, self).__init__()

        # Initial conditions
        self.x_i = torch.tensor(X_i, requires_grad=True, 
                                            dtype=torch.float32, device=device)
        self.t_i = torch.zeros_like(self.x_i, requires_grad=True)
        # Boundary conditions
        self.t0 = torch.tensor(T, requires_grad=True, 
                                            dtype=torch.float32, device=device)
        self.x0 = torch.zeros_like(self.t0, requires_grad=True)
        self.t1 = torch.tensor(T, requires_grad=True, 
                                            dtype=torch.float32, device=device)
        self.x1 = torch.ones_like(self.t1, requires_grad=True)
        # Collocation points
        self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True, 
                                            dtype=torch.float32, device=device)
        self.t_f = torch.tensor(X_f[:, 1:2], requires_grad=True, 
                                            dtype=torch.float32, device=device)
        self.f = torch.tensor(F, dtype=torch.float32, device=device)

        self.dnn = DNN(layer_sizes).to(device)
        self.C = C 
        self.num_iter = 0

        # self.optimizer = torch.optim.LBFGS(
        #     self.dnn.parameters(),
        #     lr = 1.,
        #     max_iter=50000,
        #     max_eval=50000,
        #     history_size=50,
        #     tolerance_grad=1e-16,
        #     tolerance_change=1.0 * np.finfo(float).eps,
        #     line_search_fn='strong_wolfe'
        # )

        self.optimizer = torch.optim.Adam(
            self.dnn.parameters(),
            lr = 0.1,
        )
        self.max_num_iter = 100
        

    def net_u(self, x, t):
        u = self.dnn(torch.cat((x, t), dim=1))
        return u 

    def net_f(self, x, t, f):
        u = self.net_u(x, t)

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
        # Loss f
        Lu = self.net_f(self.x_f, self.t_f, self.f)
        print("Max lu %.4e"% (Lu.max().item()))
        loss_f = torch.mean(torch.square(Lu))
        # Loss i
        u = self.net_u(self.x_i, self.t_i)
        u_t = torch.autograd.grad(
            u, self.t_i,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        loss_i = torch.mean(torch.square(u)) + torch.mean(torch.square(u_t))
        # Loss b
        u0 = self.net_u(self.x0, self.t0)
        u0_t = torch.autograd.grad(
            u0, self.t0,
            grad_outputs=torch.ones_like(u0),
            retain_graph=True,
            create_graph=True
        )[0]
        u0_x = torch.autograd.grad(
            u0, self.x0,
            grad_outputs=torch.ones_like(u0),
            retain_graph=True,
            create_graph=True
        )[0]
        loss_l = torch.mean(torch.square(u0_t / self.C - u0_x))
        u1 = self.net_u(self.x1, self.t1)
        u1_t = torch.autograd.grad(
            u1, self.t1,
            grad_outputs=torch.ones_like(u1),
            retain_graph=True,
            create_graph=True
        )[0]
        u1_x = torch.autograd.grad(
            u1, self.x1,
            grad_outputs=torch.ones_like(u1),
            retain_graph=True,
            create_graph=True
        )[0]
        loss_r = torch.mean(torch.square(u1_t / self.C + u1_x))
        loss_b = loss_l + loss_r
        loss = loss_f + loss_i + loss_b
        return loss, loss_f, loss_i, loss_b

    def train(self):
        self.dnn.train()
        while self.num_iter < self.max_num_iter:
            self.optimizer.zero_grad()
            loss, loss_f, loss_i, loss_b = self.compute_loss()
            loss.backward()
            self.num_iter += 1
            if self.num_iter % 1 == 0:
                print('Iter %d, Loss: %.5e, Loss_f:%.5e, Loss_i:%.5e, Loss_b:%.5e' 
                % (self.num_iter, loss.item(), loss_f.item(), loss_i.item(), loss_b.item()))
            self.optimizer.step()
            
    def predict(self, X, F):
        x = torch.tensor(X[:, 0:1], requires_grad=True, device=device, dtype=torch.float32)
        t = torch.tensor(X[:, 1:2], requires_grad=True, device=device, dtype=torch.float32)
        f = torch.tensor(F, dtype=torch.float32, device=device) 
        self.dnn.eval()   
        u = self.net_u(x, t)
        Lu = self.net_f(x, u, f)
        u = u.detach().cpu().numpy()
        Lu = Lu.detach().cpu().numpy()
        return u, Lu

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
    return w# *1e2

def compute_f(x, t, x0, f0):
    """RHS(Nonhomogenenous term) of wave equaion 
    f(x, t) = w(t)*delta(x-x0)
    """
    d = delta(x, x0)
    w = ricker(t, f0)
    f = w * d
    return f

def gen_data(ni=1000, nb=1000, nf=10000):
    X_i = np.linspace(0, 1, ni).reshape(-1, 1)
    T = np.linspace(0, 3, nb).reshape(-1, 1)
    X_f = np.random.rand(nf, 2)
    X_f[:, 1] = X_f[:, 1] * 3
    F = compute_f(X_f[:, 0], X_f[:, 1], x0=0.1, f0=10.).reshape(-1, 1)
    return X_i, T, X_f, F 

def draw_predict(model):
    data = np.loadtxt('data/1dwave.txt')
    X = data[:, :2]
    u_star = data[:, 2]
    F = compute_f(X[:, 0], X[:, 1], x0=0.1, f0=10.).reshape(-1, 1)
    n = len(X)
    u = np.zeros((n, 1))
    Lu = np.zeros((n, 1))
    space = 100
    idx = np.arange(0, n, space)
    for i in range(len(idx)-1):
        ui, Lui = model.predict(X[idx[i]:idx[i+1]], F[idx[i]:idx[i+1]])
        u[idx[i]:idx[i+1]] = ui 
        Lu[idx[i]:idx[i+1]] = Lui
    ui, Lui = model.predict(X[idx[-1]:], F[idx[-1]:])
    u[idx[-1]:] = ui
    Lu[idx[-1]:] = Lui
    mse = np.mean(np.square(u - u_star[:, np.newaxis]))
    res = np.mean(np.square(Lu))
    print('The MSE is %.4e, mean residual is %.4e' % (mse, res))
    u = u.reshape(3600, 201)
    fig = plt.imshow(u, cmap="gray", aspect="auto")
    plt.title('Predict')
    plt.savefig("figures/1dwave_pinn.png")

if __name__ == '__main__':
    X_i, T, X_f, F = gen_data()
    layer_sizes = [2] + [128] * 5 + [1]
    pinn = PINN(X_i, T, X_f, F, layer_sizes, C=1)
    pinn.train()
    # draw_predict(pinn)
    # x = torch.rand(100, 1, device=device)
    # t = torch.rand(100, 1, device=device)
    # out = pinn.net_u(x, t)

