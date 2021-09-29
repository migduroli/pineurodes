import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np


torch.set_default_dtype(torch.float64)

class Lorenz(nn.Module):

    def __init__(self, sigma=10., beta=8./3., rho=28.):
        super().__init__()

        self.sigma = float(sigma)
        self.beta = float(beta)
        self.rho = float(rho)

    def forward(self, t, y):
        x, y, z = torch.unbind(y)

        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z

        dLdt = torch.stack([dxdt, dydt, dzdt])
        return dLdt


t = torch.arange(0.0, 37.5, 0.005)
initial_state = torch.Tensor([1., 1., 1.])

sigma = 10.
beta = 8./3.
rho = 28.

with torch.no_grad():
    solution = odeint(Lorenz(sigma, beta, rho), initial_state, t).numpy()

tpoints = t
xpoints = solution[:, 0]
ypoints = solution[:, 1]
zpoints = solution[:, 2]

equationdata = np.array(list(zip(tpoints, xpoints, ypoints, zpoints)))

np.savetxt('Lorenz_data.txt', equationdata)
