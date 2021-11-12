import os
import time
import numpy as np
import numpy.random as npr
import argparse
import json
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"


parser = argparse.ArgumentParser('Eyam Learning')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=113)
parser.add_argument('--batch_time', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=4000)
parser.add_argument('--test_freq', type=int, default=100)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

torch.set_default_dtype(torch.float64)

sir_y0 = torch.Tensor([254, 7, 0]).to(device)
sir_t = torch.arange(0.0, 124, 1).to(device)

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

with torch.no_grad():
    sir_y = odeint(SIR(alpha, beta), sir_y0, sir_t, method='dopri5')
np.savetxt('SIR.txt', sir_y)


t_eyam = torch.Tensor([0, 15, 30, 46, 61, 77, 92, 123])
s_eyam = torch.Tensor([254, 235, 201, 153.5, 121, 108, 97, 83])
i_eyam = torch.Tensor([7, 14.5, 22, 29, 20, 8, 8, 0])
r_eyam = torch.Tensor([0, 11.5, 38, 78.5, 120, 145, 156, 178])

s_itp = interp1d(t_eyam, s_eyam, kind='slinear')
i_itp = interp1d(t_eyam, i_eyam, kind='slinear')
r_itp = interp1d(t_eyam, r_eyam, kind='slinear')

t = torch.arange(0., 124., 1).to(device)

s_new = torch.Tensor(s_itp(t))
i_new = torch.Tensor(i_itp(t))
r_new = torch.Tensor(r_itp(t))

true_y0 = torch.Tensor([254, 7, 0]).to(device)
true_y = torch.stack((s_new, i_new, r_new), 0).to(device)
true_y = torch.transpose(true_y, 0, 1).type(torch.Tensor)

def get_batch(train_size, batch_size, batch_time, true_y, t):
    s = torch.from_numpy(np.random.choice(np.arange(train_size - 1.25*batch_time, dtype=np.int64), batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)
    test_y0 = true_y[s+batch_time]
    test_t = t[:int(batch_time/4)]
    test_y = torch.stack([true_y[s+batch_time + i] for i in range(int(batch_time/4))], dim=0)# (T, M, D)
    return batch_y0, batch_t, batch_y, test_y0, test_t, test_y

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')
    fig = plt.figure(figsize=(10, 4), facecolor='white')
    ax = fig.add_gridspec(1, 10)
    ax1 = fig.add_subplot(ax[0, 0:5])
    ax2 = fig.add_subplot(ax[0, 6:])

def visualize(time, true_y, pred_y, itr):

    if args.viz:
        ax1.cla()
        ax1.plot(t_eyam, s_eyam, 'bo', t_eyam, i_eyam, 'ro', t_eyam, r_eyam, 'go', )
        ax1.plot(sir_t.numpy(), sir_y.numpy()[:, 0], 'b', sir_t.numpy(), sir_y.numpy()[:, 1], 'r', sir_t.numpy(),
                 sir_y.numpy()[:, 2], 'g')
        ax1.plot(t.numpy(), pred_y.numpy()[:, 0], 'b--', t.numpy(), pred_y.numpy()[:, 1], 'r--', t.numpy(),
                 pred_y.numpy()[:, 2], 'g--')
        # ax1.legend(loc=1)
        ax1.set_xlabel('Days', fontsize=10)
        ax1.set_ylabel('Numbers', fontsize=10)
        ax1.set_xlim(-5, 129)
        ax1.set_ylim(-10, 260)
        ax1.text(-25, 250, '(a)', fontsize=15)

        ax2.cla()
        ax2.plot(s_eyam, i_eyam, 'ko', label='Original data')
        ax2.plot(sir_y.numpy()[:, 0], sir_y.numpy()[:, 1], 'k', label='SIR model')
        ax2.plot(pred_y.numpy()[:, 0], pred_y.numpy()[:, 1], 'k--', label='Prediction')
        ax2.set_xlabel("Susceptible", fontsize=10)
        ax2.set_ylabel("Infected", fontsize=10)
        ax2.set_xlim(75, 260)
        ax2.set_ylim(-1, 31)
        ax2.text(43, 29.5, '(b)', fontsize=15)

        plt.savefig('png/{}.png'.format(itr))

class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 9),
            nn.ReLU(),
            nn.Linear(9, 9),
            nn.ReLU(),
            nn.Linear(9, 3)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)

def saveStateDict(input, itr):
    """
    A function used to save model.state_dict() OrderedDict as json file with '%s.json'%itr file name.
    """
    input_dict = dict()
    for k, v in input.items():
        input_dict[k] = v.numpy().tolist()
    if not os.path.exists('Learning'):
        os.makedirs('Learning')
    with open('Learning/{}.json'.format(itr), 'w') as fp:
        json.dump(input_dict, fp)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

if __name__ == '__main__':


    func = ODEFunc().to(device)

    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)

    loss_train_list = ([])
    loss_test_list = ([])
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y, test_y0, test_t, test_y = get_batch(args.data_size, args.batch_size, args.batch_time, true_y, t)
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()


        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())
        loss_train = loss/4
        if itr % args.test_freq == 0: # or itr == 1:
            with torch.no_grad():
                pred_test = odeint(func, test_y0, test_t)
                loss_test = torch.mean(torch.abs(pred_test - test_y))
                #pred_train = odeint(func, true_y0, t[:len(train_y)])
                #loss_train = torch.mean(torch.abs(pred_train - train_y))
                print('Iter {:04d} | Train loss {:.6f} | Validation loss {:.6f}'.format(itr, loss_train, loss_test))

                pred = odeint(func, true_y0, t)
                makedirs('Dataset_cv')
                np.savetxt('Dataset_cv\{}.txt'.format(itr), pred)
                visualize(t, true_y, pred, itr)
                #visualize(t, train_y, pred_train, itr)
            loss_train_list.append([itr, loss_train])
            loss_test_list.append([itr, loss_test])
            saveStateDict(func.state_dict(), itr)
        end = time.time()
        np.savetxt('loss_train_cv.txt', loss_train_list)
        np.savetxt('loss_test_cv.txt', loss_test_list)

loss_train_data = np.loadtxt('loss_train_cv.txt')
loss_test_data = np.loadtxt('loss_test_cv.txt')
plt.figure()
plt.plot(loss_train_data[:, 0], loss_train_data[:, 1], label='train loss')
plt.plot(loss_test_data[:, 0], loss_test_data[:, 1], label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('loss_graph_cv.png')

# find corresponding coefficient matrix
ind = np.argmin(loss_train_data, axis=0)
print('The minimum loss:', loss_train_data[ind[1]])