import sympy
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import numpy as np

import torch
import torch.nn as nn

from torchdiffeq import odeint

import json
import argparse


parser = argparse.ArgumentParser('SIR_result')
parser.add_argument('--itr', type=int, default=400)
args = parser.parse_args()

with open('Learning/{}.json'.format(args.itr)) as json_file:
    data = json.load(json_file)

weight = []
bias = []

for i in range(0, 7, 2):
    weight.append(data["net.{}.weight".format(i)])
    bias.append(data["net.{}.bias".format(i)])

class Eyam(nn.Module):

    def __init__(self, weight, bias):
        super(Eyam, self).__init__()
        self.weight = weight
        self.bias = bias
        self.relu = nn.ReLU()

    def forward(self, t, y):
        y = torch.matmul(torch.Tensor(self.weight[0]), y) + torch.Tensor(self.bias[0])
        for i in range(1, 4):
            y = self.relu(y)
            y = torch.matmul(torch.Tensor(self.weight[i]), y) + torch.Tensor(self.bias[i])
        return y

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

alpha = 2.82/31
beta = alpha/159

sir_y0 = torch.Tensor([254, 7, 0])

t = torch.arange(0., 200.)
initial_state = torch.Tensor([254., 7., 0])


with torch.no_grad():
    pred_y = odeint(Eyam(weight, bias), initial_state, t)
    sir_y = odeint(SIR(alpha, beta), sir_y0, t, method='dopri5')


t_eyam = torch.Tensor([0, 15, 30, 46, 61, 77, 92, 123])
s_eyam = torch.Tensor([254, 235, 201, 153.5, 121, 108, 97, 83])
i_eyam = torch.Tensor([7, 14.5, 22, 29, 20, 8, 8, 0])
r_eyam = torch.Tensor([0, 11.5, 38, 78.5, 120, 145, 156, 178])



fig = plt.figure(figsize=(10, 4), facecolor='white')
ax = fig.add_gridspec(1, 10)
ax1 = fig.add_subplot(ax[0, 0:5])
ax2 = fig.add_subplot(ax[0, 6:])

ax1.cla()
ax1.plot(t_eyam, s_eyam, 'bo', t_eyam, i_eyam, 'ro', t_eyam, r_eyam, 'go', )
ax1.plot(t.numpy(), sir_y.numpy()[:, 0], 'b', t.numpy(), sir_y.numpy()[:, 1], 'r', t.numpy(),
         sir_y.numpy()[:, 2], 'g')
s_pred = pred_y.numpy()[:, 0]
i_pred = pred_y.numpy()[:, 1]
r_pred = pred_y.numpy()[:, 2]
tot_pred = s_pred+i_pred+r_pred
ax1.plot(t.numpy(), pred_y.numpy()[:, 0], 'b--', t.numpy(), pred_y.numpy()[:, 1], 'r--', t.numpy(),
         261-pred_y[:, 0]-pred_y[:, 1], 'g--')
         #pred_y.numpy()[:, 2], 'g--')

# ax1.legend(loc=1)
ax1.set_xlabel('Days', fontsize=10)
ax1.set_ylabel('Numbers', fontsize=10)
#ax1.set_xlim(-5, 200)
ax1.set_ylim(-10, 275)
ax1.text(-40, 250, '(a)', fontsize=15)

ax2.cla()
ax2.plot(s_eyam, i_eyam, 'ko', label='Original data')
ax2.plot(sir_y.numpy()[:, 0], sir_y.numpy()[:, 1], 'k', label='SIR model')
ax2.plot(pred_y.numpy()[:, 0], pred_y.numpy()[:, 1], 'k--', label='Prediction')
ax2.set_xlabel("Susceptible", fontsize=10)
ax2.set_ylabel("Infected", fontsize=10)
ax2.set_xlim(75, 260)
ax2.set_ylim(-1, 31)
ax2.text(43, 29.5, '(b)', fontsize=15)


plt.savefig("Eyam_result.png")
plt.show()
