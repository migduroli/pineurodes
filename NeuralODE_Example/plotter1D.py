import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
    plt.figure(figsize=figsize)
    plt.title('1D Linear Case', fontsize=30)
    plt.xlabel('t', fontsize=20)
    plt.ylabel('x', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    if obs is not None:
        if times is None:
            times = [None] * len(obs)
        for o, t in zip(obs, times):
            o, t = to_np(o), to_np(t)
            for b_i in range(o.shape[1]):
                plt.plot(o[:, b_i, 0], 'ro')


    if trajs is not None:
        for z in trajs:
            z = to_np(z)
            plt.plot(z[:, 0, 0], lw=1.5)
        if save is not None:
            plt.savefig(save)
    plt.show(block=False)
    #plt.pause(0.5)
    plt.close()