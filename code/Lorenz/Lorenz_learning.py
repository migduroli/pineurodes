import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import json

parser = argparse.ArgumentParser('Lorenz Learning')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=4000)
parser.add_argument('--batch_time', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--vc_init_flag', type=int, default=10)
parser.add_argument('--ode_nums', type=int, default=3)
parser.add_argument('--basis_order', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--test_freq', type=int, default=100)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()
vail_init_value = None

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
torch.manual_seed(3)
equationdata = np.loadtxt('Lorenz_data.txt')
true_y = torch.from_numpy(equationdata[:, 1:].reshape(-1, 1, args.ode_nums)).type(torch.FloatTensor)
true_yorig = true_y
true_y0 = true_y[0]
t = torch.from_numpy(equationdata[:, 0]).type(torch.FloatTensor)

# Set data_size and train_size
args.data_size = t.size()[0]
args.train_size = int(args.data_size * 0.8)
train_y = true_y[:args.train_size]
test_y = true_y[args.train_size:]


def get_batch(train_size, batch_size, batch_time, true_y, t):
    s = torch.from_numpy(np.random.choice(np.arange(train_size - batch_time, dtype=np.int64), batch_size, replace=False))
    batch_y0 = train_y[s]  # (M, D)
    batch_t = t[:batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt
    fig = plt.figure(facecolor='white')
    ax_traj = fig.add_subplot(311)
    ay_traj = fig.add_subplot(312)
    az_traj = fig.add_subplot(313)



def visualize(time, true_y, pred_y, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_ylabel('x', fontsize=15)
        ax_traj.plot(time.numpy(), true_y.numpy()[:, 0, 0], 'b-', time.numpy(), pred_y.numpy()[:, 0, 0], 'r-')
        ax_traj.set_xlim(time.min(), time.max())


        ay_traj.cla()
        ay_traj.set_ylabel('y', fontsize=15)
        ay_traj.plot(time.numpy(), true_y.numpy()[:, 0, 1], 'b-', time.numpy(), pred_y.numpy()[:, 0, 1], 'r-', )
        ay_traj.set_xlim(time.min(), time.max())

        az_traj.cla()
        az_traj.set_ylabel('z', fontsize=15)
        az_traj.plot(time.numpy(), true_y.numpy()[:, 0, 2], 'b-', time.numpy(), pred_y.numpy()[:, 0, 2], 'r-')
        az_traj.set_xlim(time.min(), time.max())


        fig.tight_layout()
        plt.savefig('png/{}.png'.format(itr))
        #plt.draw()
        #plt.pause(0.001)

class OdeNet(nn.Module):
    def __init__(self, ode_nums, basis_order, vc_init_flag):
        super(OdeNet, self).__init__()
        self._total_basis = int(np.math.factorial(ode_nums+basis_order)/(np.math.factorial(ode_nums)*np.math.factorial(basis_order)))
        self.vc = nn.Parameter(torch.Tensor(self._total_basis, ode_nums))
        self.ode_nums = ode_nums
        self.basis_order = basis_order

        self._vc_init_flag = vc_init_flag
        self.vc.data.uniform_(-2, 2)
        vail_init_value = self.vc.data.numpy()
        #self.vc.data[0,:] = 0

    def forward(self, t, x_input):

        def _compute_theta3d():
            _basis_count = 0
            _Theta = torch.zeros(x_input.size(0), 1, self._total_basis)
            _Theta[:, 0, 0] = 1
            _basis_count += 1
            for ii in range(0, self.ode_nums):
                _Theta[:, 0, _basis_count] = x_input[:, 0, ii]
                _basis_count += 1

            if self.basis_order >= 2:
                for ii in range(0, self.ode_nums):
                    for jj in range(ii, self.ode_nums):
                        _Theta[:, 0, _basis_count] = torch.mul(x_input[:, 0, ii], x_input[:, 0, jj])
                        _basis_count += 1

            if self.basis_order >= 3:
                for ii in range(0, self.ode_nums):
                    for jj in range(ii, self.ode_nums):
                        for kk in range(jj, self.ode_nums):
                            _Theta[:, 0, _basis_count] = torch.mul(torch.mul(x_input[:, 0, ii], x_input[:, 0, jj]), x_input[:, 0, kk])
                            _basis_count += 1

            assert _basis_count == self._total_basis
            return _Theta

        def _compute_theta2d():
            _basis_count = 0
            _Theta = torch.zeros(x_input.size(0), self._total_basis)
            _Theta[:, 0] = 1
            _basis_count += 1
            for ii in range(0,self.ode_nums):
                _Theta[:, _basis_count] = x_input[:, ii]
                _basis_count += 1

            if self.basis_order >= 2:
                for ii in range(0, self.ode_nums):
                    for jj in range(ii, self.ode_nums):
                        _Theta[:, _basis_count] = x_input[:, ii]*x_input[:, jj]
                        _basis_count += 1

            if self.basis_order >= 3:
                for ii in range(0, self.ode_nums):
                    for jj in range(ii, self.ode_nums):
                        for kk in range(jj, self.ode_nums):
                            _Theta[:, _basis_count] = x_input[:, ii]*x_input[:, jj]*x_input[:, kk]
                            _basis_count += 1

            assert _basis_count == self._total_basis
            return _Theta

        def _vc_init():
            _left_dxdt = torch.rand(x_input.size(0), x_input.size(2))*2
            #_left_dxdt = _left_dxdt.reshape(-1,self.ode_nums)
            #_left_dxdt = _left_dxdt - torch.from_numpy(np.roll(_left_dxdt,1,axis=0))
            _Theta_init = _compute_theta3d().reshape(-1, self._total_basis)
            return torch.mm(torch.from_numpy(np.linalg.pinv(_Theta_init)), _left_dxdt)

        if self._vc_init_flag == 0:
            self.vc.data = _vc_init()
            #self.vc.data.uniform_(-1, 1)
            self._vc_init_flag = 1
            vail_init_value = self.vc.data.numpy()

        if x_input.dim() == 2:
            output = torch.mm(_compute_theta2d(), self.vc)
        else:
            output = torch.matmul(_compute_theta3d(), self.vc)

        return output

def saveStateDict(input, itr):
    """
    A function used to save model.state_dict() OrderedDict as json file with 'matrix%s.json'%itr file name.
    """
    input_dict = dict()
    for k, v in input.items():
        input_dict[k] = v.numpy().tolist()
    if not os.path.exists('Learning'):
        os.makedirs('Learning')
    with open('Learning/{}.json'.format(itr), 'w') as fp:
        json.dump(input_dict, fp)

class ShrinkParameter(object):

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, module):
        if hasattr(module, 'vc'):
            w = module.vc.data
            #print(module.vc)
            _indexzero = abs(w)/abs(w).max() < self.threshold
            module.vc.data[_indexzero] = 0

class Shrink(object):

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, module, threshold):
        self.threshold = threshold
        if hasattr(module, 'vc'):
            w = module.vc.data
            _indexzero = abs(w) < self.threshold
            module.vc.data[_indexzero] = 0
            for p in module.parameters():
                p.grad[_indexzero] = 0


def adjust_learning_rate(optimizer, itr):
    """Sets the learning rate to the initial LR decayed by 0.5 every 1000 epochs"""
    lr = args.lr * (0.9 ** (itr // 1000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def initialize_parameters(func, batch_y, batch_t, ode_nums, basis_order, vc_init_flag):
    if vc_init_flag == 0:
        pass
    else:
        batch_y_st = batch_y[0:-1, 0, 0, :]
        batch_y_et = batch_y[1:, 0, 0, :]
        dy = batch_y_et - batch_y_st
        dt = batch_t[1]-batch_t[0]
        _dydt = dy/dt

        _basis_count = 0
        _total_basis = int(np.math.factorial(ode_nums+basis_order)/(np.math.factorial(ode_nums)*np.math.factorial(basis_order)))
        _Theta = torch.zeros(batch_y.size(0)-1, _total_basis)
        _Theta[:, 0] = 1
        _basis_count += 1
        for ii in range(0, ode_nums):
            _Theta[:, _basis_count] = batch_y_st[:, ii]
            _basis_count += 1

        if basis_order >= 2:
            for ii in range(0, ode_nums):
                for jj in range(ii, ode_nums):
                    _Theta[:, _basis_count] = batch_y_st[:, ii]*batch_y_st[:, jj]
                    _basis_count += 1

        if basis_order >= 3:
            for ii in range(0, ode_nums):
                for jj in range(ii, ode_nums):
                    for kk in range(jj, ode_nums):
                        _Theta[:, _basis_count] = batch_y_st[:, ii]*batch_y_st[:, jj]*batch_y_st[:, kk]
                        _basis_count += 1

        assert _basis_count == _total_basis

        func.vc.data = torch.mm(torch.from_numpy(np.linalg.pinv(_Theta)), _dydt)



if __name__ == '__main__':

    ii = 0

    func = OdeNet(args.ode_nums,args.basis_order,args.vc_init_flag)
    func.float()
    # 0.005 0.001

    shrinkparameter = ShrinkParameter(0.0001)
    shrink = Shrink(0.0001)
    w = list(func.parameters())

    #optimizer = optim.RMSprop(func.parameters(), lr=args.lr)
    optimizer = optim.Adam(func.parameters(), lr=args.lr, betas=(0, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #optimizer = optim.SGD(func.parameters(), lr = args.lr)
    end = time.time()

    # The batch structure:
    #--#---#-----#----- batch_y0=[#,#,#]
    #__#*@-#*@---#*@---- batch_y or pred_y = [[#,#,#],[*,*,*],[@,@,@]]
    #Where the number of # is batch_size, the number type of symbols  is batch_time. Here they are 3 both in
    #this scheme.

    loss_train_list = list([])
    loss_test_list = list([])
    loss_batch_list = list([])
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        #print(func.vc)
        batch_y0, batch_t, batch_y = get_batch(args.train_size, args.batch_size, args.batch_time, true_y, t)
        batch_y0 = batch_y0.type(torch.FloatTensor)
        batch_t = batch_t.type(torch.FloatTensor)
        batch_y = batch_y.type(torch.FloatTensor)
        # if flag = 1, optimize by outer function, elif flag = 0, optimize by inner function
        if itr == 1:
            _, batch_ti, batch_yi = get_batch(args.train_size, args.batch_size, args.train_size-args.batch_size-args.batch_time, true_yorig, t)
            initialize_parameters(func, batch_yi, batch_ti, args.ode_nums, args.basis_order, args.vc_init_flag)
            func.apply(shrinkparameter)
        pred_y = odeint(func, batch_y0, batch_t)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        #print(itr, "==> loss: ", loss.item())
        # print('loss')
        # print(loss)
        l1_regularization = torch.tensor(0).type(torch.FloatTensor)
        reg_loss = 0
        for param in func.parameters():
            l1_regularization += torch.norm(param,1)
        # 0.05
        factor = 0.01 if 0.005*1.2**(itr/1000) > 0.01 else 0.005*1.3**(itr/1000)
        #factor = 0.00001*0.5**(itr/500)
        loss += factor * l1_regularization

        loss_batch_list.append([itr,loss.data.item()])
        #if itr == 10000:
            #np.savetxt('loss_batch{}.txt'.format(args.CASE), loss_batch_list)

        #print(itr)
        if torch.isnan(func.vc.max()):
            print('Non')
        #print(func.vc)
        func.vc.data[0, :] = 0
        #func.vc.data[1, 2] = 0
        #func.vc.data[2, 2] = 0
        #func.vc.data[3, :2] = 0
        #func.vc.data[4, :] = 0
        #func.vc.data[5, :2] = 0
        #func.vc.data[6, 0] = 0
        #func.vc.data[6, 2] = 0
        #func.vc.data[7:, :] = 0
        loss.backward()
        #if itr<5000:
        #func.apply(shrink)
        shrink(func,0.001 if 0.0001*2*itr/1000>0.001 else 0.0005*2**(itr/1000))
        adjust_learning_rate(optimizer, itr)
        optimizer.step()






        if itr % args.test_freq == 0 or itr == 1:
            with torch.no_grad():
                pred_test = odeint(func, test_y[0], t[len(train_y):])
                loss_test = torch.mean(torch.abs(pred_test - test_y))
                #print('Iter {:04d} | test Loss {:.6f}'.format(itr, loss_test.item()))
                pred_train = odeint(func, true_y0, t[:len(train_y)]) ###########################,method='rk4'
                loss_train = torch.mean(torch.abs(pred_train - train_y))/4
                print('Iter {:04d} | train Loss {:.6f} | Validationt loss {:6f}'.format(itr, loss_train.item(), loss_test.item()))
                # update args.test_freq
                if loss_train > 3:
                    args.test_freq = 500
                    args.lr = 0.1
                elif loss_train > 2:
                    args.test_freq = 100
                    args.lr = 0.05
                elif loss_train > 1:
                    args.test_freq = 10
                    args.lr = 0.01
                else:
                    args.lr = 0.001
                    args.test_freq = 1
                # save (itr,loss) pair
                loss_test_list.append([itr, loss_test.data.item()])
                np.savetxt('loss_test.txt', loss_test_list)
                loss_train_list.append([itr, loss_train.data.item()])
                np.savetxt('loss_train.txt', loss_train_list)

                #visualize(t[len(train_y):], test_y, pred_test, func, itr, loss_test.item())
                visualize(t[:len(train_y)], train_y, pred_train, itr)

            print(func.vc)

                #pred_all_y = odeint(func, true_y0, t)
                #visualize(t, true_y, pred_all_y, func, ii)

                #Write the parameters to json file
            saveStateDict(func.state_dict(), itr)
        end = time.time()
    #print(vail_init_value)

# plot and save loss_itr figure

loss_train_data=np.loadtxt('loss_train.txt')
loss_test_data=np.loadtxt('loss_test.txt')
plt.figure()
plt.plot(loss_train_data[:,0],loss_train_data[:, 1], label='train')
plt.plot(loss_test_data[:,0],loss_test_data[:, 1], label='test')
plt.xlabel('itr')
plt.ylabel('total train set loss')
plt.legend()
plt.savefig('loss_graph.png')

# find corresponding coefficient matrix
ind = np.argmin(loss_train_data, axis=0)
print('The minimum loss:', loss_train_data[ind[1]])

