
"""
This is support code for creating complex valued neural nets.
"""

import os
from abc import ABC
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch as t
from torch.autograd import Function
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from itertools import cycle, count
_, _, _, _, _, _ = optim, SummaryWriter, DataLoader, DataLoader, TensorDataset, Dataset
_, _, _ = train_test_split, cycle, count

# ========================== Complex Activation Functions ====================================


def crelu(re, im):
    return nn.functional.relu(re), nn.functional.relu(im)


def zrelu(re, im):
    abs_ = t.sqrt(re ** 2 + im ** 2)
    ang = t.atan2(im, re)
    ang[~(ang > 0)] = 0
    return abs_ * t.cos(ang), abs_ * t.sin(ang)


def modrelu(re, im, alpha):
    abs_ = t.sqrt(re ** 2 + im ** 2)
    ang = t.atan2(im, re)
    abs_ = nn.functional.relu(abs_ + alpha)
    return abs_ * t.cos(ang), abs_ * t.sin(ang)


def zlogit(re, im):
    abs_ = t.sqrt(re ** 2 + im ** 2)
    ang = t.atan2(im, re)
    mask = ~(ang > 0)
    clone = abs_.clone()
    clone[mask] = -1 * abs_[mask]
    return clone


# =============================== Complex Layers =================================================


class Clinear(nn.Module, ABC):
    def __init__(self, in_, out_):
        super(Clinear, self).__init__()
        self.Lre = nn.Linear(in_, out_)
        self.Lim = nn.Linear(in_, out_)

    def forward(self, re, im):
        return self.Lre(re) - self.Lim(im), self.Lre(im) + self.Lim(re)


class Cconv1d(nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super(Cconv1d, self).__init__()
        self.Lre = nn.Conv1d(*args, **kwargs)
        self.Lim = nn.Conv1d(*args, **kwargs)

    def forward(self, re, im):
        return self.Lre(re) - self.Lim(im), self.Lre(im) + self.Lim(re)


class Cconv2d(nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super(Cconv2d, self).__init__()
        self.Lre = nn.Conv2d(*args, **kwargs)
        self.Lim = nn.Conv2d(*args, **kwargs)

    def forward(self, re, im):
        return self.Lre(re) - self.Lim(im), self.Lre(im) + self.Lim(re)


# ============================ Adverserial Training =========================================


class ReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)  # or:  x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # lambda_ = grads.new_tensor(lambda_) # To make sure that lambda has the same dtype as the gradients
        output = grad_output.neg() * ctx.alpha
        return output, None


#%% ================================= Loss Functions ======================================


def cluster_dist(prediction, dd):
    """Experimental function to measure how clustery a preduction is. The intuition is that we want to
    make use of geometry of the stroke predicted and the ground truth. In particular, we aim at discouraging
    the neural net from making a prediction where the neural net throws away a bunch of strokes all over
    the place. Thus, we want to measure how close are the grid cells and whether they form a cluster indeed.
    Shoud be capable of working with batches of predictions.
    :param prediction:
    :type prediction:
    :param dd:
    :return:
    :rtype:
    after some training:
    prediction = my_net(x_source_test)
    x = ((prediction.unsqueeze(1) * D).sum(dim=2) * prediction).sum(dim=1)
    plt.plot(x.cpu().detach().numpy())
    """
    return ((prediction.unsqueeze(1) * dd).sum(dim=2) * prediction).sum(dim=1).mean()


def cxe(predicted, target):
    """ Categorical Cross Entropy loss.
    The problem with this is that after some time, predicted may be reduced to zero at some points leading to nan
    Clipping is recommended for anything inside the log
    :param predicted:
    :type predicted:
    :param target:
    :type target:
    :return:
    :rtype:
    """
    return -(target * t.log(predicted.clamp_min(1e-7))).sum(dim=1).mean()


def my_kl(predicted, target):
    """
    So the problem with Categorical Cross entropy is that sometimes even though the prediction is so close to the
    target, CXE still spits out very large number, especially when compared with other case in which the prediction
    was poorly resemblant of the target. The reason behind this is  CXE is measuring average message length, and
    according to Gibs inequality, this is always larger than the entropy of target distribution. This means that
    if we have a case in which the target distribution has large entropy, then, even if the prediction was very
    good, the average message length will still have to exceed that value of auto-entropy. To solve this, let's
    subtract the entropy of the target from the CXE result. Well, that's just re-inventing the KL divergence
    which is literally a measure of the information gain.
    KL = AVG message length when diverging from P to Q - Entropy of P.
    It makes much more sense to attempt to make two distributions similar in case of soft-classes classification
    This is in contrast to regular classfication problem where we seek something different.

    :param predicted:
    :type predicted:
    :param target:
    :type target:
    :return:
    :rtype:
    """
    return -(target * t.log(predicted.clamp_min(1e-7))).sum(dim=1).mean() - \
           -1 * (target.clamp(min=1e-7) * t.log(target.clamp(min=1e-7))).sum(dim=1).mean()


#%% ======================= NN ultis =========================================


class Accuracy(object):
    """This is a memic of Keras Accuracy class for Pytorch.
    """
    def __init__(self):
        self.counter = 0.0
        self.correct = 0.0

    def __call__(self, y_pred, y_true):
        with t.no_grad():
            _, predicted = t.max(y_pred.data, dim=1)
            corr = (predicted == y_true).sum().item()
        self.counter += len(y_true)
        self.correct += corr
        return corr / len(y_true)

    def result(self):
        return 100 * self.correct / self.counter

    def reset_states(self):
        self.counter = 0.0
        self.correct = 0.0


class Reshape(nn.Module, ABC):
    def __init__(self):
        super(Reshape, self).__init__()

    def forward(self, x):
        _ = self
        return x.view()


def tb(pn=8888):
    with open('batch.bat', 'w') as file:
        file.write(f'tensorboard --logdir ./runs --host localhost --port {pn}')
    os.startfile('batch.bat')


def conv_output_shape(h_w=(30, 91), kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(((h_w[0] + (2 * padding) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = floor(((h_w[1] + (2 * padding) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


#%% ========================= Brain Injury Vizulaization tools ==================================


class Viz:
    def __init__(self):
        plt.style.use('fivethirtyeight')  # dark_background
        plt.switch_backend('qt5agg')  # Necessary for figure window maximizer and manager.
        self.focus = np.load('./data/source/processed/mask_for_meshing.npy')[45:105, 50:100]
        self.cells = np.load('./data/source/processed/meshcells.npy')
        self.ax = None

    @staticmethod
    def latext():
        plt.style.use('default')  # dark_background
        from matplotlib import rc
        rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        # rc('text', usetex=True)
        size = 35
        plt.rc('xtick', labelsize=size)
        plt.rc('ytick', labelsize=size)
        plt.rc('axes', labelsize=size)
        # plt.rc('title', labelsize=size)
        plt.rc('legend', fontsize=size-10)

    def viz(self, predicted_label, correct_label, update=True):
        """
        To visulaize one instance of predicted and correct distribution of classes.
        :param predicted_label:
        :type predicted_label: torch.Tensor
        :param correct_label:
        :type correct_label: torch.Tensor
        :param update: Do you want to update the previous shown figure, or create a new one?
        :type update: bool
        :return: A figure
        :rtype: plt.figure
        """
        current_figs = plt.get_figlabels()
        if ('viz_distribution' in current_figs) and update:  # No need to create a new fig
            fig = plt.figure('viz_distribution')  # Choose that figure
            ax = fig.axes
            for an_ax in ax:
                an_ax.cla()  # Clear it
        else:  # If there wasn't a figure, means this is first call, or, update is False ==>
            # both cases: create a new one
            _ = plt.figure(num='viz_distribution')
            ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
            ax2 = plt.subplot2grid((2, 3), (0, 2))
            ax3 = plt.subplot2grid((2, 3), (1, 2))
            ax = [ax1, ax2, ax3]
        # In any case, do the maximization
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()  # Works only for Qt backend
        self.ax = ax
        # ----------------- Plotting the distributions --------------------------------------------------
        width = 0.25  #
        ax[0].set_title(f'CXE = {cxe(predicted_label.unsqueeze(0), correct_label.unsqueeze(0)):1.3f}, '
                        f'KLD = {my_kl(predicted_label.unsqueeze(0), correct_label.unsqueeze(0)):1.3f}')
        predicted_label, correct_label = predicted_label.cpu().detach().numpy(), correct_label.cpu().detach().numpy()
        ax[0].bar(np.arange(len(predicted_label)) - width, correct_label, width=width, label='Target Distribution')
        ax[0].bar(np.arange(len(predicted_label)), predicted_label, width=width, label='Predicted Distribution')
        ax[0].set_xticks(np.arange(len(predicted_label)))
        ax[0].set_xticklabels(np.arange(len(predicted_label)), size=12)
        ax[0].legend()
        ax[0].set_xlabel('classes')
        ax[0].set_ylabel('distribution')
        # ---------------- Plotting the brain -------------------------------------------------------------
        labels = [predicted_label, correct_label]
        string = ['Predicted label', 'Correct label']
        h, w = 7, 7
        for i in range(1, 3, 1):
            alabel = labels[i - 1]
            ax[i].imshow(self.focus * 1.0)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            ax[i].set_title(string[i - 1])
            for index, acell in enumerate(self.cells):
                x0, y0 = acell
                rect = mpatches.Rectangle([x0, y0], w, h, color='r', alpha=alabel[index], ec="green")
                # Thre rect takes [x, y] as coordinates and extends right and down. Meaning Top left is origin
                ax[i].add_patch(rect)
                ax[i].text(x0 + w / 4, y0 + h / 2, str(index), size=10)

    def add_image(self, image_path=None):
        """Replaces the actual distribution with an image.
        """
        if image_path is None:
            tmp = r'G:\emvdata\MeasurementData\\\20190528-GlyWater-CNC-AA-AS\\TargetDia37\\Photos\\CNC-Exp0032.jpg'
            image_path = tmp
        image = plt.imread(image_path)
        self.ax[1] = plt.imshow(image[300: 1450, 1600: 2500])
        plt.xticks([])
        plt.yticks([])
        plt.title('Real label', fontsize=10)
        # plt.savefig(f'{results_path}result.png', bbox_inches='tight', pad_inches=0.5)


def evaluate_model(model, x, y, data='source_test', number=20):
    """
    data can be: 'source_test', 'source_train', 'target_train', 'target_test'
    """
    with t.no_grad():
        res = model(x)
        try:
            predicted_labels, _ = res
        except TypeError:
            predicted_labels = res
    loss = my_kl(predicted_labels, y)
    print(f'average loss = {loss.item()}')
    # Selecting samples to plot
    if number == 0:
        start = 0
        number = len(y)
    else:
        start = np.random.randint(len(y) - number)

    myviz = Viz()
    for i in range(start, start + number, 1):
        myviz.viz(predicted_labels[i], y[i])
        plt.xlabel(f'{data}_{i}')
        plt.pause(0.02)
        plt.savefig(f'./temp/{data}_{start}_{i}.png')


def get_magnitude(ref):  # absing
    # Converting x_source and x_target to ABS instead of [re, im] channels.
    # This is useful to see the information gain in frquencies alone, rather than complex values.
    tmp = abs(np.real(ref[:, 0, :, :]) + 1j * np.imag(ref[:, 1, :, :]))
    return tmp[:, None, :, :]  # Maintain a dim of 1 for channels.
