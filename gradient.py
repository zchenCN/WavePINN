"""
Layers for computing gradients of wave equations

@data: 2022-05-11
"""
import torch 
import torch.nn as nn


class GradientLayer(nn.Module):
    """
    Custom layer to compute 1st and 2nd derivatives for the wave equation.

    Attributes:
        model: pytorch network model.
    """

    def __init__(self, model, **kwargs):
        """
        Args:
            model: keras network model.
        """
        super().__init__(**kwargs)
        self.model = model


    def forward(self, *args):
        """
        Computing 1st and 2nd derivatives for the wave equation.

        Args:
            *args: input variables (t, x) for 1d and (t, x, z) for 2d.

        Returns:
            u: network output.
            du_dt: 1st derivative of t.
            du_dx: 1st derivative of x.
            du_dz: 1st derivative of z
            d2u_dt2: 2nd derivative of t.
            d2u_dx2: 2nd derivative of x.
            d2u_dz2: 2nd derivative of z
        """
        assert all([isinstance(x, torch.Tensor) for x in args])
        dim = len(args) - 1
        if dim == 1:
            t, x = args 
        elif dim == 2:
            t, x, z = args 
        else:
            raise ValueError('Number of arguments should be 2 or 3')
        u = self.model(torch.cat(args, dim=1))
        du_dt = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        d2u_dt2 = torch.autograd.grad(
            du_dt, t, 
            grad_outputs=torch.ones_like(du_dt),
            retain_graph=True,
            create_graph=True
        )[0]
        du_dx = torch.autograd.grad(
            u, x, 
            torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        d2u_dx2 = torch.autograd.grad(
            du_dx, x, 
            torch.ones_like(du_dx),
            retain_graph=True,
            create_graph=True
        )[0]
        if dim == 1:
            return u, du_dt, du_dx, d2u_dt2, d2u_dx2
        elif dim == 2:
            du_dz = torch.autograd.grad(
                u, z, 
                torch.ones_like(u),
                retain_graph=True,
                create_graph=True
            )[0]
            d2u_dz2 = torch.autograd.grad(
                du_dz, z, 
                torch.ones_like(du_dz),
                retain_graph=True,
                create_graph=True
            )[0]
            return u, du_dt, du_dx, du_dz, d2u_dt2, d2u_dx2, d2u_dz2