import numpy as np
from scipy.io import loadmat
import pandas as pd
from glob import glob
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib import rc


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

# CXE VS KLD: # See code @ V1-AllFreqConv2D, last cell

# NonSat formulation of loss: See code @ PaperResults / plots / last cell
# Vanilla MLP vs Multi-freq vs CV-NN See code @ PaperResults / plots
# Softbinning See code @ Source Prepocessing in Headscanner Project, last cell
# Simulation results vs Real data: See code @ Magnitude Processing in HeadScanner Project: last cell
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
style = 'Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap,' \
        ' CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd,' \
        ' OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r,' \
        ' Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r,' \
        ' PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu,' \
        ' RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r,' \
        ' Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr,' \
        ' YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r,' \
        ' bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm,' \
        ' coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, ' \
        'gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r,' \
        ' gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r,' \
        ' gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r,' \
        ' inferno, inferno_r, jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r,' \
        ' ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r,' \
        ' seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r,' \
        ' tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, twilight, twilight_r,' \
        ' twilight_shifted, twilight_shifted_r, viridis, viridis_r, winter, winter_r'.replace(' ', '').split(',')
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


plt.style.use('default')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
size = 20
plt.rc('xtick', labelsize=size)
plt.rc('ytick', labelsize=size)
plt.rc('axes', labelsize=size)
plt.rc('legend', fontsize=15)
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

import toolbox as tb
tmp = r"G:\emvision\Head_Phantoms\Fabricated\20180709-liquidphantomto test the system-BM\bloodtargetmaterls.csv"
blood = tb.pd.read_csv(tmp, skiprows=12)
tmp = r"G:\emvision\Head_Phantoms\Fabricated\20180709-liquidphantomto test the system-BM\liquidphantom-test-system.csv"
phantom = tb.pd.read_csv(tmp, skiprows=12)
colors = tb.Cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

fig, ax1 = plt.subplots(figsize=(11, 8))
color = 'tab:red'
ax1.set_xlabel('Frequency (GHz)')
ax1.set_ylabel('Relative Permittivity')
# ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Conductivity (S/m)')
ax1.plot(blood['frequency(Hz)'] / 1e9, blood["e'"], color=colors.next(),
         label='Blood emulating fluid permittivity')
ax1.plot(phantom['frequency(Hz)'] / 1e9, phantom["e'"], color=colors.next(),
         label='Brain emulating fluid permittivity')
ax2.plot(blood['frequency(Hz)'] / 1e9, blood["e''/e'"] * blood["e'"] * 2 * np.pi * blood['frequency(Hz)'] * 8.851e-12,
         color=colors.next(), label='Blood emulating fluid conductivity')
ax2.plot(phantom['frequency(Hz)'] / 1e9, phantom["e''/e'"] * phantom["e'"] * 2 * np.pi * phantom['frequency(Hz)'] * 8.851e-12,
         color=colors.next(), label='Brain emulating fluid conductivity')
ax1.legend(loc='lower right')
ax2.legend(loc='best')
tb.FigureManager.grid(ax1)
fig.savefig("props.png", bbox_inches='tight', pad_inches=0.5)
