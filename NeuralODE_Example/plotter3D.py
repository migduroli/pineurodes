import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
import seaborn as sns

sns.color_palette("bright")


def to_np(x):
    return x.detach().cpu().numpy()


def plot_trajectories(obs=None, times=None, trajs=None, save=None, figsize=(16, 8)):
    """
    plotter for the trajectory
    :param obs: true observation
    :param times: all time step
    :param trajs: predicted trajectory
    :param save: bool
    :param figsize:
    :return: None
    """
    mpl.rcParams['font.size'] = 15
    plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
    plt.title('3D Linear Case', fontsize=30)
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('z')

    if obs is not None:
        if times is None:
            times = [None] * len(obs)
        for o, t in zip(obs, times):
            o, t = to_np(o), to_np(t)
            for b_i in range(o.shape[1]):
                #plt.scatter(o[:, b_i, 0], o[:, b_i, 1], o[:, b_i, 2], c=t[:, b_i, 0], cmap=cm.plasma)
                plt.plot(o[:, b_i, 0], o[:, b_i, 1], o[:, b_i, 2], 'o')

    if trajs is not None:
        for z in trajs:
            z = to_np(z)
            plt.plot(z[:, 0, 0], z[:, 0, 1], z[:, 0, 2], lw=1.5)
        if save is not None:
            plt.savefig(save)
    plt.show(block=False)
    #plt.pause(0.5)
    plt.close()