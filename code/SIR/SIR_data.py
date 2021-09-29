import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt


class SIR(nn.Module):

    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, t, y):
        s, i, r = torch.unbind(y)

        ds_dt = -self.beta * s * i
        di_dt = self.beta * s * i - self.alpha * i
        dr_dt = self.alpha * i
        return torch.stack([ds_dt, di_dt, dr_dt])

alpha = 0.1
beta = 0.001

t = torch.arange(0.0, 50., 0.1)
true_y0 = torch.Tensor([997., 3., 0])


with torch.no_grad():
    true_y = odeint(SIR(alpha, beta), true_y0, t)
    true_y = true_y + torch.randn_like(true_y) * 20

susceptible = true_y[:, 0]
infected = true_y[:, 1]
recovered = true_y[:, 2]

data = np.array(list(zip(t, susceptible, infected, recovered)))

np.savetxt('SIR_data.txt', data)

fig = plt.figure()
plt.ylim(-20, 1020)
plt.plot(susceptible[0:-1:10], 'bo', markersize=3, label='Susceptible')
plt.plot(infected[0:-1:10], 'ro', markersize=3, label='Infected')
plt.plot(recovered[0:-1:10], 'go', markersize=3, label='Recovered')
plt.xlabel("Days")
plt.ylabel("Numbers")
plt.legend()
plt.savefig("SIR_data.png")
plt.show()