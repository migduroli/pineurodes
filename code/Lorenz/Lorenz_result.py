import sympy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import torch
import torch.nn as nn

from torchdiffeq import odeint

import json
import argparse

import seaborn as sns

sns.color_palette("bright")
plt.rcParams["font.family"] = "Times New Roman"

parser = argparse.ArgumentParser('Lorenz_result')
parser.add_argument('--itr', type=int, default=260)
args = parser.parse_args()

with open('Learning/{}.json'.format(args.itr)) as json_file:
    data = json.load(json_file)

parameter_matrix = np.matrix(data["vc"])
x, y, z, xx, xy, xz, yy, yz, zz = sympy.symbols('x, y, z, xx, xy, xz, yy, yz, zz')
variable_matrix = sympy.Matrix([1, x, y, z, xx, xy, xz, yy, yz, zz]).reshape(1, 10)
result = variable_matrix * parameter_matrix

print("===========Learned Lorenz System===========")
print("dx/dt = {}".format(result[0]))
print("dy/dt = {}".format(result[1]))
print("dz/dt = {}".format(result[2]))



class Lorenz(nn.Module):

    def __init__(self, matrix):
        super().__init__()

        self.parameters0 = matrix[:, 0]
        self.parameters1 = matrix[:, 1]
        self.parameters2 = matrix[:, 2]

    def forward(self, t, y):
        x, y, z = torch.unbind(y)

        matrix_xyz = np.array([1, x, y, z, x*x, x*y, x*z, y*y, y*z, z*z])

        dxdt = torch.tensor(matrix_xyz * self.parameters0)[0, 0]
        dydt = torch.tensor(matrix_xyz * self.parameters1)[0, 0]
        dzdt = torch.tensor(matrix_xyz * self.parameters2)[0, 0]
        dLdt = torch.stack([dxdt, dydt, dzdt])
        return dLdt

t = torch.arange(0.0, 30, 0.005)
initial_state = torch.Tensor([1., 1., 1.])


with torch.no_grad():
    learnedLorenz = odeint(Lorenz(parameter_matrix), initial_state, t).numpy()

originalLorenz = np.loadtxt("Lorenz_data.txt")


fig = plt.figure(figsize=(8, 8))
ax = fig.gca(projection='3d')
plt.gca().patch.set_facecolor('white')
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.set_xlabel('x', fontsize=17)
ax.set_ylabel('y', fontsize=17)
ax.set_zlabel('z', fontsize=17)

ax.plot(originalLorenz[0:6000:10, 1], originalLorenz[0:6000:10, 2], originalLorenz[0:6000:10, 3], 'bo', markersize=1)
ax.plot(learnedLorenz[:, 0], learnedLorenz[:, 1], learnedLorenz[:, 2], 'r')
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='z', labelsize=14)


plt.savefig("lorenz_attractor.png")
plt.show()

fig = plt.figure(figsize=(8, 8))
plt.subplot(3,1,1)
plt.plot(t, originalLorenz[:6000, 1], 'b', t, learnedLorenz[:, 0], 'r')
plt.ylabel('x')
plt.subplot(3,1,2)
plt.plot(t, originalLorenz[:6000, 2], 'b', t, learnedLorenz[:, 1], 'r')
plt.ylabel('y')
plt.subplot(3,1,3)
plt.plot(t, originalLorenz[:6000, 3], 'b', t, learnedLorenz[:, 2], 'r')
plt.ylabel('z')
plt.savefig("lorenz_xyz.png")
plt.show()
