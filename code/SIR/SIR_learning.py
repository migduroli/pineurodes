import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser('SIR_learning')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=50)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=1500)
parser.add_argument('--ode_nums', type=int, default=3)
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
equationdata = np.loadtxt('SIR_data.txt')
true_y = torch.from_numpy(equationdata[:, 1:].reshape(-1, 1, args.ode_nums)).type(torch.FloatTensor)
true_yorig = true_y
true_y0 = true_y[0]
t = torch.from_numpy(equationdata[:, 0]).type(torch.FloatTensor)
# Set data_size and train_size
args.data_size = t.size()[0]
args.train_size = int(args.data_size * 0.8)
train_y = true_y[:args.train_size]
test_y = true_y[args.train_size:]


def get_batch():
    s = torch.from_numpy(
        np.random.choice(np.arange(args.train_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = train_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([train_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')
    fig = plt.figure(figsize=(11, 4.4), facecolor='white')
    ax = fig.add_gridspec(1, 10)
    ax1 = fig.add_subplot(ax[0, 0:5])
    ax2 = fig.add_subplot(ax[0, 6:])

def visualize(time, true_y, pred_y, itr):
    if args.viz:
        ax1.cla()
        ax1.plot(time.numpy()[0:-1:10], true_y.numpy()[0:-1:10, 0, 0], 'bo', label='Original Susceptible', markersize=2)
        ax1.plot(time.numpy()[0:-1:10], pred_y.numpy()[0:-1:10, 0, 0], 'b--', label='Prediction Susceptible')
        ax1.plot(time.numpy()[0:-1:10], true_y.numpy()[0:-1:10, 0, 1], 'ro', label='Original Infected', markersize=2)
        ax1.plot(time.numpy()[0:-1:10], pred_y.numpy()[0:-1:10, 0, 1], 'r--', label='Prediction Infected')
        ax1.plot(time.numpy()[0:-1:10], true_y.numpy()[0:-1:10, 0, 2], 'go', label='Original Recovered', markersize=2)
        ax1.plot(time.numpy()[0:-1:10], pred_y.numpy()[0:-1:10, 0, 2], 'g--', label='Prediction Recovered')
        # ax1.legend(loc=1)
        ax1.set_xlabel('Days', fontsize=14)
        ax1.set_ylabel('Numbers', fontsize=14)
        ax1.set_xlim(-2, 52)
        ax1.set_ylim(-20, 1100)
        ax1.text(-12, 1040, '(a)', fontsize=20)
        ax1.tick_params(axis='x', labelsize=13)
        ax1.tick_params(axis='y', labelsize=13)

        ax2.cla()
        ax2.plot(true_y.numpy()[0:-1:10, 0, 0], true_y.numpy()[0:-1:10, 0, 1], 'ko', label='Original', markersize=2)
        ax2.plot(pred_y.numpy()[0:-1:10, 0, 0], pred_y.numpy()[0:-1:10, 0, 1], 'k--', label='Prediction')
        ax2.set_xlabel("Susceptible", fontsize=14)
        ax2.set_ylabel("Infected", fontsize=14)
        ax2.set_ylim(-50, 790)
        ax2.text(-320, 745, '(b)', fontsize=20)
        ax2.tick_params(axis='x', labelsize=13)
        ax2.tick_params(axis='y', labelsize=13)

        plt.savefig('png/{}.png'.format(itr))

class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 9),
            nn.ReLU(),
            nn.Linear(9, 27),
            nn.ReLU(),
            nn.Linear(27, 50),
            nn.ReLU(),
            nn.Linear(50, 27),
            nn.ReLU(),
            nn.Linear(27, 9),
            nn.ReLU(),
            nn.Linear(9, 3)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)



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
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()


        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0 or itr == 1:
            with torch.no_grad():
                pred_test = odeint(func, test_y[0], t[len(train_y):])
                loss_test = torch.mean(torch.abs(pred_test - test_y))
                pred_train = odeint(func, true_y0, t[:len(train_y)])
                loss_train = torch.mean(torch.abs(pred_train - train_y)) / 4
                print('Iter {:04d} | Train loss {:.6f} | Validation loss {:.6f}'.format(itr, loss_train, loss_test))

                pred = odeint(func, true_y0, t)
                visualize(t, true_y, pred, itr)
                #visualize(t, train_y, pred_train, itr)
            loss_train_list.append([itr, loss_train])
            loss_test_list.append([itr, loss_test])

        end = time.time()
        np.savetxt('loss_train.txt', loss_train_list)
        np.savetxt('loss_test.txt', loss_test_list)

loss_train_data = np.loadtxt('loss_train.txt')
loss_test_data = np.loadtxt('loss_test.txt')
plt.figure()
plt.plot(loss_train_data[:, 0], loss_train_data[:, 1], label='train loss')
plt.plot(loss_test_data[:, 0], loss_test_data[:, 1], label='validation loss')
plt.xlabel('itr')
plt.ylabel('loss')
plt.savefig('loss_graph.png')

# find corresponding coefficient matrix
ind = np.argmin(loss_train_data, axis=0)
print('The minimum loss:', loss_train_data[ind[1]])
