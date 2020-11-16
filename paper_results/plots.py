import numpy as np
from scipy.io import loadmat
import pandas as pd
from glob import glob
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib import rc
import toolbox as tb


#%% Setting up matplotlib
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
size = 20
plt.rc('xtick', labelsize=size)
plt.rc('ytick', labelsize=size)
plt.rc('axes', labelsize=size)
plt.rc('legend', fontsize=15)
# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.unicode'] = True
# plt.style.use('fivethirtyeight')

#%%

results_path = rf".\tmp"
path_to_data = rf'.\plots_data\real_vs_complex'
files = glob(path_to_data + '/*')
data = []
for afile in files:
    data.append(pd.read_csv(afile))

# %% If you want to mix plots with different x-axis.

plt.close('all')
fig = plt.figure()
curve1 = ax1 = fig.add_subplot(111, label='Test set loss')
ax1.plot(data[0]['Step'], data[0]['Value'], color='C0', linewidth=2, label='Training loss')
ax1.set_xlabel('Batches')
ax1.tick_params(axis='x', colors="C0")
ax1.set_ybound(0, 2)
ax1.grid(False, axis='x')
ax1.grid(True, axis='y')

ax2 = fig.add_subplot(111, label='Training set loss', frame_on=False)
curve2 = ax2.plot(data[1]['Step'], data[1]['Value'], color='C1', linewidth=2, label='Test loss')
ax2.set_xlabel('Epochs')
ax2.set_yticks([])
ax2.tick_params(axis='x', colors="C1")
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.xaxis.set_label_position('top')
ax2.yaxis.set_label_position('right')
ax2.set_ybound(0, 2)

# fig.legend(loc=(0.8, 0.7))

# %%
plt.close('all')
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

_ = ax1.plot(data[0]['Step'].values, data[0]['Value'].values, color='C0', linewidth=2, label='Training loss')
ax1.set_xlabel('Batches')
ax1.tick_params(axis='x', colors="C0")
ax1.set_ybound(0, 2)
ax1.grid(False, axis='x')
ax1.grid(True, axis='y')

_ = ax2.plot(data[1]['Step'].values, data[1]['Value'].values, color='C1', linewidth=2, label='Test loss')
ax2.set_xlabel('Epochs')
ax2.set_yticks([])
ax2.tick_params(axis='x', colors="C1")
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.xaxis.set_label_position('top')
ax2.yaxis.set_label_position('right')
ax2.set_ybound(0, 2)

# ax1.legend([curve1, curve2], ['e', 'e'])


# %% Vanilla MLP vs Complex MLP. AllDataReal1DConv vs AllFreqAllDataComplex1Dconv
plt.close('all')
_, ax = plt.subplots(dpi=100)
fig.set_size_inches(10, 10)

labels = ['RV-MLP', 'CV-MLP', 'RV-1D-CNN', '30F-CV-1D-CNN']
for i in range(len(labels)):
    y = data[i]['Value']
    y = savgol_filter(y, 51, 3)[:-5]  # window size 51, polynomial order 3
    x = data[i]['Step'][:-5]
    ax.semilogy(x, y, linewidth=2, label=labels[i])
ax.set_title('\\textbf{Losses on different architectures}')
ax.legend()
ax.set_xlabel('batch')
ax.set_ylabel('Loss')
ax.set_yticks(np.logspace(np.log(0.3), np.log(1), 10))
ax.set_ylim([0, 1])
ax.set_yticklabels(np.round(np.logspace(np.log(0.3), np.log(1), 10), 2))
ax.set_xticklabels(ax.get_xticks().astype(int))
ax.grid(True, which='major')

for aspine in list(ax.spines.keys()):
    ax.spines[aspine].set_color('black')
    ax.spines[aspine].set_linewidth(2)
fig.savefig(results_path + 'loss.png')

# %% Plot for nonsaturated DC formulation

plt.close('all')
size = 35
x = np.arange(0.01, 0.99, 0.01)
y2 = -np.log(1 - x)
y1 = -np.log(x)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
ax[0].plot(x, y1, linewidth=2)
ax[0].set_title('Loss for instances with label "1"', fontsize=size)
ax[0].set_ylabel('Loss', fontsize=size)
ax[0].set_xlabel('A: confidence of label being "1"', fontsize=size)
ax[0].grid()
ax[0].axvspan(0.5, 1, color='red', alpha=0.1)
ax[0].axvspan(0, 0.5, color='green', alpha=0.1)
ax[0].set_xticklabels(np.round(ax[0].get_xticks(), 1), size=size)
ax[0].set_yticklabels(np.round(ax[0].get_yticks(), 1), size=size)

ax[1].plot(x, y2, linewidth=2)
ax[1].set_title('Loss for instances with label "0"', fontsize=size)
# ax[1].set_ylabel('Loss', fontsize=size)
ax[1].set_xlabel('B: confidence of label being "1"', fontsize=size)
ax[1].grid()
ax[1].axvspan(0.5, 1, color='green', alpha=0.1)
ax[1].axvspan(0, 0.5, color='red', alpha=0.1)
ax[1].set_xticklabels(np.round(ax[1].get_xticks(), 1), size=size)
ax[1].set_yticklabels(np.round(ax[1].get_yticks(), 1), size=size)
fig.savefig(results_path + 'nonsat.png', layout='tight', pad_inches=0.1, bbox_inches='tight')


# %% # Real data strokes.
path_to_15mm = r'G:\emvision\Platform\MeasurementData\20190528-GlyWater-CNC-AA-AS\TargetDia15\Photos'
path_to_37mm = r'G:\emvision\Platform\MeasurementData\20190528-GlyWater-CNC-AA-AS\TargetDia37\Photos'
file1 = glob(path_to_15mm + '/*')[np.random.randint(100)]
file2 = glob(path_to_37mm + '/*')[np.random.randint(100)]
im1 = plt.imread(file1)
im2 = plt.imread(file2)

plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 2

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), dpi=100,
                       facecolor=(0.9411764705882353, 0.9411764705882353, 0.9411764705882353, 1.0))
im = [im1, im2]
for an_ax, an_im in zip(ax, im):
    an_ax.imshow(an_im[300: 1450, 1400: 2600])
    an_ax.grid(False)
    an_ax.set_xticks([])
    an_ax.set_yticks([])
# plt.tight_layout()
fig.savefig(f'{results_path}reality_sample.png', transparent=False, bbox_inches='tight', pad_inches=0.5)

# %% Simulation stroke Sample
"""Simluation Results from FDTD codes
For this purpose, we might as well use pics from original dataset which come with
higher resolution (300 * 300) instead of the dataset that we really worked with (150 * 150)"""

epr_database, sig_database = None, None

path_to_data = r'.\ml_stroke_localisation-master\data'
files = glob(path_to_data + '/*')
for a_file in files:
    data = loadmat(a_file)
    exec(list(data.keys())[-1] + '=' + 'data[list(data.keys())[-1]]')
    name = list(data.keys())[-1]
    print(name, eval(name).shape)

i = 1
style = plt.colormaps()
N = len(style)
fig = plt.figure(figsize=(20, 900))
counter = 0
for k in range(N):
    ax = fig.add_subplot(N, 6, k+1)
    ax.imshow(sig_database[:, :, i][90: 210, 90: 210]/sig_database[:, :, i][90: 210, 90: 210].max(),
              cmap=style[counter])
    ax.set_xlabel(style[counter] + '  ' + str(counter))
    ax.set_xticks([])
    ax.set_yticks([])
    counter += 1

N = 4
M = 2
r = 800
cmap = 'gist_earth'
fig, ax = plt.subplots(M, N, figsize=(13, 5),
                       facecolor=(0.9411764705882353, 0.9411764705882353, 0.9411764705882353, 1.0))
for j in range(N):
    imag = epr_database[:, :, j + r][90: 210, 90: 210]  # /epr_database[:, :, j+r][90: 210, 90: 210].max()
    plot = ax[0, j].imshow(imag, cmap=cmap)
    ax[0, j].set_xticks([])
    ax[0, j].set_yticks([])
    cbar = fig.colorbar(plot, ax=ax[0, j])
    cbar.ax.tick_params(labelsize=10)
ax[0, 0].set_ylabel('Permitivity', size=size)
for j in range(N):
    imag = sig_database[:, :, j + r][90: 210, 90: 210] + 0.05  # /epr_database[:, :, j+r][90: 210, 90: 210].max()
    plot = ax[1, j].imshow(imag, cmap=cmap)
    ax[1, j].set_xticks([])
    ax[1, j].set_yticks([])
    cbar = fig.colorbar(plot, ax=ax[1, j])
    cbar.ax.tick_params(labelsize=10)

ax[1, 0].set_ylabel('Conductivity', size=size)
fig.suptitle('', size=15)
fig.set_dpi(150)
fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
fig.set_facecolor('none')

fig.savefig(f'{results_path}simulations_sample.png', bbox_inches='tight', pad_inches=0.5)

print(f'{results_path}simulations_sample.png')

#%% Plotting properties of blood and brain.

tmp = r"G:\emvision\Head_Phantoms\Fabricated\20180709-liquidphantomto test the system-BM\bloodtargetmaterls.csv"
blood = tb.pd.read_csv(tmp, skiprows=12)
tmp = r"G:\emvision\Head_Phantoms\Fabricated\20180709-liquidphantomto test the system-BM\liquidphantom-test-system.csv"
phantom = tb.pd.read_csv(tmp, skiprows=12)
colors = tb.Cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
blood["e'"] = savgol_filter(blood["e'"], window_length=51, polyorder=3)
blood["e''/e'"] = savgol_filter(blood["e''/e'"], window_length=51, polyorder=3)
phantom["e'"] = savgol_filter(phantom["e'"], window_length=51, polyorder=3)
phantom["e''/e'"] = savgol_filter(phantom["e''/e'"], window_length=51, polyorder=3)

fig, ax1 = plt.subplots(figsize=(11, 8))
color = 'tab:red'
ax1.set_xlabel('Frequency (GHz)')
ax1.set_ylabel('Relative Permittivity')
# ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Conductivity (S/m)')
step = 8
x = blood['frequency(Hz)'] / 1e9
y = blood["e'"]
x = x[::step]
y = y[::step]
ax1.plot(x, y, color=colors.next(), linestyle=':', marker='^',
         label='Blood emulating fluid permittivity')
x = phantom['frequency(Hz)'] / 1e9
y = phantom["e'"]
x = x[::step]
y = y[::step]
ax1.plot(x, y, color=colors.next(), linestyle='--', marker='x',
         label='Brain emulating fluid permittivity')
x = blood['frequency(Hz)'] / 1e9
y = blood["e''/e'"] * blood["e'"] * 2 * np.pi * blood['frequency(Hz)'] * 8.851e-12
x = x[::step]
y = y[::step]
ax2.plot(x, y, linestyle='-.', marker='s',
         color=colors.next(), label='Blood emulating fluid conductivity')
x = phantom['frequency(Hz)'] / 1e9
y = phantom["e''/e'"] * phantom["e'"] * 2 * np.pi * phantom['frequency(Hz)'] * 8.851e-12
x = x[::step]
y = y[::step]
ax2.plot(x, y, linestyle='-.', marker='o',
         color=colors.next(), label='Brain emulating fluid conductivity')
ax1.legend(loc='lower right')
ax2.legend(loc='best')
tb.FigureManager.grid(ax1)
fig.savefig("paper_results/resources/props.png", bbox_inches='tight', pad_inches=0.5)

#%% Others

# CXE VS KLD: # See code @ V1-AllFreqConv2D, last cell
# Softbinning See code @ Source Prepocessing, last cell
# Simulation results vs Real data: See code @ Magnitude: last cell


