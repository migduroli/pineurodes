import torch
import torch.nn as nn

from torchdiffeq import odeint

import numpy as np

import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

class LotkaVolterra(nn.Module):
    def __init__(self, a, alpha, b, beta):
        super(LotkaVolterra, self).__init__()
        self.a = a
        self.alpha = alpha
        self.b = b
        self.beta = beta

    def forward(self, t, y):
        # x = prey, y = predator
        x, y = torch.unbind(y)

        dx_dt = self.a * x - self.alpha * x * y
        dy_dt = -self.b * y + self.beta * x * y
        return torch.Tensor([dx_dt, dy_dt])


t = torch.arange(0.0, 10.0, 0.1)
true_y0 = torch.Tensor([1., 1.])

a = 1.5
alpha = 1
b = 3
beta = 1

with torch.no_grad():
    true_y = odeint(LotkaVolterra(a, alpha, b, beta), true_y0, t)
    true_y = true_y + torch.randn_like(true_y) * 0.2

prey = true_y[:, 0]
predator = true_y[:, 1]

data = np.array(list(zip(t, prey, predator)))

np.savetxt('LV_data.txt', data)

plt.figure()
plt.plot(prey, 'bo', label='prey')
plt.plot(predator, 'ro', label='predator')
plt.legend()
plt.savefig("LV_data.png")
plt.show()