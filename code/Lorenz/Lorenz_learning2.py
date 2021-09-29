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
import json

parser = argparse.ArgumentParser('Lorenz_learning')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=3000)
parser.add_argument('--fold', type=int, default=5)
parser.add_argument('--niters', type=int, default=5)
parser.add_argument('--vc_init_flag', type=int, default=1)
parser.add_argument('--ode_nums', type=int, default=3)
parser.add_argument('--basis_order', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--test_freq', type=int, default=1)
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
equationdata = np.loadtxt('Lorenz_data.txt')
true_y = torch.from_numpy(equationdata[:, 1:].reshape(-1, 1, args.ode_nums)).type(torch.FloatTensor)
true_yorig = true_y
true_y0 = true_y[0]
t = torch.from_numpy(equationdata[:, 0]).type(torch.FloatTensor)

args.data_size = t.size()[0]


def get_batch(data_size, fold, true_y, t, i_f):
    fold_length = int(data_size / fold)
    batch_y0 = true_y[0]
    batch_t = t[:fold_length * i_f]
    batch_y = torch.stack([true_y[i] for i in range(fold_length*i_f)], dim=0)
    test_y0 = true_y[fold_length * i_f]
    test_t = t[:fold_length]
    test_y = torch.stack([true_y[fold_length*i_f + i] for i in range(fold_length)], dim=0)
    return batch_y0, batch_t, batch_y, test_y0, test_t, test_y



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



def visualize(time, true_y, pred_y, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_ylabel('x')
        ax_traj.plot(time.numpy(), true_y.numpy()[:, 0, 0], 'b-', time.numpy(), pred_y.numpy()[:, 0, 0], 'r-')
        ax_traj.set_xlim(time.min(), time.max())
        #ax_traj.legend()


        ay_traj.cla()
        ay_traj.set_ylabel('y')
        ay_traj.plot(time.numpy(), true_y.numpy()[:, 0, 1], 'b-', time.numpy(), pred_y.numpy()[:, 0, 1], 'r-')
        ay_traj.set_xlim(time.min(), time.max())
        # ax_traj.set_ylim(-2, 2)
        #ay_traj.legend()

        az_traj.cla()
        az_traj.set_ylabel('z')
        az_traj.plot(time.numpy(), true_y.numpy()[:, 0, 2], 'b-', time.numpy(), pred_y.numpy()[:, 0, 2], 'r-')
        az_traj.set_xlim(time.min(), time.max())
        # ax_traj.set_ylim(-2, 2)
        #az_traj.legend()


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
        batch_y_st = batch_y[0:-1, 0, :]
        batch_y_et = batch_y[1:, 0, :]
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

    func = OdeNet(args.ode_nums, args.basis_order, args.vc_init_flag)
    func.float()
    # 0.005 0.001
    shrinkparameter = ShrinkParameter(0.001)
    shrink = Shrink(0.01)
    w = list(func.parameters())

    #optimizer = optim.RMSprop(func.parameters(), lr=args.lr)
    optimizer = optim.Adam(func.parameters(), lr=args.lr, betas=(0, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #optimizer = optim.SGD(func.parameters(), lr = args.lr)

    # The batch structure:
    #--#---#-----#----- batch_y0=[#,#,#]
    #__#*@-#*@---#*@---- batch_y or pred_y = [[#,#,#],[*,*,*],[@,@,@]]
    #Where the number of # is batch_size, the number type of symbols  is batch_time. Here they are 3 both in
    #this scheme.

    loss_train_list = list([])
    loss_test_list = list([])
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        loss_train_set = []
        loss_test_set = []
        for i_f in range(1, args.fold):
            # batch_y0, batch_t, batch_y = get_batch(args.train_size, args.batch_size, args.batch_time, true_y, t)
            batch_y0, batch_t, batch_y, test_y0, test_t, test_y = get_batch(args.data_size, args.fold, true_y, t, i_f)
            batch_y0 = batch_y0.type(torch.FloatTensor)
            batch_t = batch_t.type(torch.FloatTensor)
            batch_y = batch_y.type(torch.FloatTensor)

            # if flag = 1, optimize by outer function, elif flag = 0, optimize by inner function
            if itr == 1:
                # _, batch_ti, batch_yi = get_batch(args.train_size, args.batch_size, args.train_size-args.batch_size-args.batch_time, true_yorig, t)
                _, batch_ti, batch_yi, _, _, _ = get_batch(args.data_size, args.fold, true_yorig, t, i_f)
                initialize_parameters(func, batch_yi, batch_ti, args.ode_nums, args.basis_order, args.vc_init_flag)
                func.apply(shrinkparameter)
            pred_y = odeint(func, batch_y0, batch_t)
            loss_train = torch.mean(torch.abs(pred_y - batch_y))

            #print(itr, "==> loss: ", loss.item())
            # print('loss')
            # print(loss)
            #l1_regularization = torch.tensor(0).type(torch.FloatTensor)
            #for param in func.parameters():
            #    l1_regularization += torch.norm(param,1)
            # 0.05
            #factor = 0.01 if 0.005*1.2**(itr/1000) > 0.01 else 0.005*1.3**(itr/1000)
            #factor = 0.00001*0.5**(itr/500)
            #loss_train += factor * l1_regularization
            loss_train_set.append(float(loss_train))


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
            loss_train.backward()
            #if itr<5000:
            #func.apply(shrink)
            shrink(func, 0.02) # if 0.0001*2*itr/1000>0.001 else 0.0005*2**(itr/1000))
            #adjust_learning_rate(optimizer, itr)
            optimizer.step()





            with torch.no_grad():
                pred_test = odeint(func, test_y0, test_t)
                loss_test = torch.mean(torch.abs(pred_test - test_y))
                loss_test_set.append(float(loss_test))

        loss_train_avg = np.sum(loss_train_set)/(args.fold-1)
        loss_test_avg = np.sum(loss_test_set)/(args.fold-1)
        loss_train_list.append([itr, loss_train_avg])
        loss_test_list.append([itr, loss_test_avg])
        np.savetxt('loss_train.txt', loss_train_list)
        np.savetxt('loss_test.txt', loss_test_list)

        if itr % args.test_freq == 0 or itr == 1:
            print('Iter {:04d} | Train Loss {:.6f} | Validation Loss {:.6f}'.format(itr, loss_train_avg, loss_test_avg))
                    # update args.test_freq
            if loss_train_avg > 5:
                args.lr = 0.05
            #elif loss_train > 3:
            #    args.lr = 0.01
            #elif loss_train > 0.1:
            #    args.lr = 0.01
            else:
                args.lr = 0.01


            pred = odeint(func, true_y0, t)
            visualize(t, true_y, pred, func, itr)


            print(func.vc)

                #pred_all_y = odeint(func, true_y0, t)
                #visualize(t, true_y, pred_all_y, func, ii)

                #Write the parameters to json file
            saveStateDict(func.state_dict(), itr)
        end = time.time()
    #print(vail_init_value)

# plot and save loss_itr figure

loss_train_data = np.loadtxt('loss_train.txt')
loss_test_data = np.loadtxt('loss_test.txt')
plt.figure()
plt.plot(loss_train_data[:, 0], loss_train_data[:, 1], label='train loss')
plt.plot(loss_test_data[:, 0], loss_test_data[:, 1], 'o', label='validation loss')
plt.xlabel('itr')
plt.ylabel('loss')
plt.legend()
plt.savefig('loss_graph.png')

# find corresponding coefficient matrix
ind = np.argmin(loss_train_data, axis=0)
print('The minimum loss:', loss_train_data[ind[1]])
