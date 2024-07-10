# Kieran Owens 2023
# Code to generate flows and maps with time-varying parameters

# Contents:
# 1. Dependencies
# 2. Global variables
# 3. Experiment function
# 4. Plotting function

##############################################################################
# 1. Dependencies
##############################################################################

# scientific computing
import os
from tqdm import tqdm
import numpy as np
import scipy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sksfa import SFA
import mdp

import pycatch22

import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge

# visualisation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern']

##############################################################################
# 2. Global variables
##############################################################################

# noise levels (1 / SNR)
NOISE = [0, 0.01, 1.0]

# window sizes
W=1000

# methods
METHODS = ["SFA2", "ESN error", "CD"]

# number of iterations per noise level
ITERS = 20

# DPI for vector graphic outputs
DPI = 500

# colour palette for visualisation
PALETTE = {
    'green': "#84b657",
    'blue': "#3fadaf",
    'orange': "#d89049",
    'purple': "#9d6cc1",
    'red': "#F6423C",
    'tan': "#FFFEF8",
    'black': 'black'}#"#cb5362"}

SHADES = {
    'green': ["#6d9c44", "#84b657", "#ABCD8E"],
    'blue': ["#308688", "#3fadaf", "#68c8ca"],
    'orange': ["#ca7a2b", "#d89049", "#e6b889"],
    'purple': ["#8348ad", "#9d6cc1", "#bc9ad5"],
    'red': ["#ea120b", "#F6423C", "#f97b77"]}

# ESN variables
ESN_DIM = 30
NB_ESN = 50

# number of characteristic distance points
NB_CD_POINTS = 100

##############################################################################
# 3. Experiment function
##############################################################################

def get_ACF_zero_crossing(X):

    # number of time series variables
    _, nb_vars = X.shape

    # compute the ACF zero crossing for each variable
    ACF_crossing = []
    for j in range(nb_vars):
        acf = [np.corrcoef(X[i:,j], X[:-i,j])[0,1] for i in range(1, 1000)]
        crossing_idx = 0
        while acf[crossing_idx] > 0 and crossing_idx < 999 - 1:
            crossing_idx += 1
        ACF_crossing.append(crossing_idx + 1)

    # return the minimum time lag
    return np.min(ACF_crossing)

def compute_rho(X, p, func=None, var=None):

    rng = np.random.default_rng(0)
    rpy.set_seed(0)
    
    # time series length and no. of variables from the T x V shape
    T, nb_vars = X.shape

    # array to collect absolute Pearson correlation values
    RHOS = np.zeros((len(NOISE), ITERS, len(METHODS) + 1))

    # experiment loop
    for j in tqdm(range(ITERS)):

        # generate ESNs
        ESN_BANK = []
        rpy.verbosity(0)
        for _ in range(NB_ESN):
            reservoir = Reservoir(ESN_DIM, lr=0.5, sr=0.9, activation='tanh')
            ridge = Ridge(ridge=1e-7)
            esn_model = reservoir >> ridge
            ESN_BANK.append(esn_model)

        # random points for use with 
        CD_points = np.zeros((NB_CD_POINTS, nb_vars))

        for i2 in range(nb_vars):
            xmin = np.min(X[:,i2])
            xmax = np.max(X[:,i2])
            xrange = xmax - xmin
            CD_points[:,i2] = rng.uniform(low=xmin - (0.1 * xrange), 
                                          high=xmax + (0.1 * xrange),
                                          size=NB_CD_POINTS)
                                                
        for i, noise in enumerate(NOISE):
        

            

            # add Gaussian observation noise
            obs_noise = rng.normal(loc=0, 
                                   scale=np.sqrt(noise)*X.std(axis=0), 
                                   size=X.shape)
            Xnoise = X + obs_noise

            ###########################
            # SFA2
            ###########################
            print(f"Iter {j}, noise {noise}, SFA2...")
            tau = get_ACF_zero_crossing(Xnoise)
            SFA_list = []
            deriv_mse_list = []
            for m in np.arange(1, 21):
                flow = (mdp.nodes.TimeFramesNode(time_frames=m, gap=tau) +
                        mdp.nodes.PolynomialExpansionNode(2) +
                        mdp.nodes.SFANode(output_dim=1))
                flow.train(Xnoise)
                SFA2 = flow(Xnoise)
                SFA_list.append(SFA2)
                deriv_mse_list.append(np.mean((SFA2[1:] - SFA2[:-1])**2))
            SFA2 = SFA_list[np.nanargmin(deriv_mse_list)]
            print(p.shape, SFA2.shape)
            diff = len(p) - len(SFA2)
            RHOS[i,j,0] += np.abs(np.corrcoef(p[diff//2:-diff//2].flatten(), 
                                                  SFA2.flatten())[0,1])
            
            ###########################
            # ESN error
            ###########################
            print(f"Iter {j}, noise {noise}, ESN error...")
            ESN_error = np.zeros((T-1, NB_ESN))
            X_train = Xnoise[:-1,:]
            Y_train = Xnoise[1:, :]

            for k in range(NB_ESN):

                esn_model = ESN_BANK[k]
                esn_model.fit(X_train, Y_train, warmup=10)
                output = esn_model.run(X_train)
                ESN_error[:,k] = (output - Y_train).mean(axis=1)
                esn_model.reset()

            F_filt = scipy.ndimage.gaussian_filter(ESN_error.mean(axis=1), sigma=1000)
            RHOS[i,j,1] += np.abs(np.corrcoef(p[1:], F_filt)[0,1])

            ###########################
            # Statistical time-series feature
            ###########################
            print(f"Iter {j}, noise {noise}, time-series feature...")

            # time points at which to evaluate window features
            time_points = np.arange(0, T - W + 1, W)

            # window indices for feature calculations
            window_indices = np.array([np.arange(t, t + W) for t in time_points])

            # downsample the parameter time series using window-based averaging
            psamp = np.array([p[idx].mean() for idx in window_indices])

            # compute statistical features
            F = np.array([func(list(Xnoise[idx, var].flatten())) for idx in window_indices])
            print(F.shape)
            RHOS[i,j,3] += np.abs(np.corrcoef(psamp, F)[0,1])


            ###########################
            # Characteristic distance
            ###########################
            print(f"Iter {j}, noise {noise}, CD...")

            F = np.zeros((T, NB_CD_POINTS))
            for t in range(T):
                F[t,:] = np.linalg.norm(CD_points - Xnoise[t,:])

            F2 = np.zeros((T//W, NB_CD_POINTS))
            for t in range(T//W):
                F2[t,:] = F[t*W:(t+1)*W,:].mean(axis=0)

            F_sfa_cd = SFA(n_components=1).fit_transform(F2)
            RHOS[i,j,2] += np.abs(np.corrcoef(psamp, F_sfa_cd[:,0])[0,1])

    return RHOS

##############################################################################
# 4. Plotting function
##############################################################################

def plot_experiment(X, p, RHOS, markersize, linewidth, fname, legend=False, text=None):

    # set up figure
    fig = plt.figure(layout="constrained", figsize=(6, 2.4))
    mosaic = """
        AC
        BC
        """
    ax_dict = fig.subplot_mosaic(mosaic,
                                width_ratios = [0.6, 1],
                                height_ratios = [1, 0.2])
    
    # plot process
    if X.shape[1] == 3:
        ax_dict["A"].plot(X[:10**4,0], X[:10**4,2], c='black', linewidth=linewidth, rasterized=True)
        ax_dict["A"].scatter(X[::10,0], X[::10,2], c=p[::10], cmap='plasma', s=markersize, alpha=0.5, rasterized=True)
        xlims = (np.min(X[:,0]), np.max(X[:,0]))
        zlims = (np.min(X[:,2]), np.max(X[:,2]))
        ax_dict["A"].arrow(xlims[0], zlims[0], (xlims[1] - xlims[0])/10, 0, linewidth=0.5)
        ax_dict["A"].text(xlims[0] + (xlims[1] - xlims[0])/9, zlims[0] - (zlims[1] - zlims[0])/90, 'x')
        ax_dict["A"].arrow(xlims[0], zlims[0], 0, (zlims[1] - zlims[0])/10, linewidth=0.5)
        ax_dict["A"].text(xlims[0] - (xlims[1] - xlims[0])/90, zlims[0] + (zlims[1] - zlims[0])/8, 'z')
    else:
        ax_dict["A"].scatter(range(X.shape[0]), X[:,0], c=p, cmap='plasma', s=markersize, rasterized=True)
        xlims = (np.min(X[:,0]), np.max(X[:,0]))
        ax_dict["A"].arrow(0, xlims[0] - (xlims[1] - xlims[0])/5, 0, (xlims[1] - xlims[0])/13, linewidth=0.5)
        ax_dict["A"].text(-1000, xlims[0] - (xlims[1] - xlims[0])/10, 'x')
        ax_dict["A"].arrow(0, xlims[0] - (xlims[1] - xlims[0])/5, 10000, 0, linewidth=0.5)
        ax_dict["A"].text(12000, xlims[0] - (xlims[1] - xlims[0])/4, 'Time')
        
    ax_dict["A"].axis('off')

    # plot time-varying parameter
    ax_dict["B"].scatter(range(len(p)), p, c=p, cmap='plasma', s=0.5, rasterized=True)
    ax_dict["B"].set_xlabel('Time')
    ax_dict["B"].set_ylabel('p', labelpad=5)
    ax_dict["B"].set_xticks([0, len(p)//2, len(p)])
    ax_dict["B"].ticklabel_format(axis='x', style='sci', scilimits=(5,5))
    ax_dict["B"].spines['top'].set_visible(False)
    ax_dict["B"].spines['right'].set_visible(False)

    # plot SFA2 data
    ax_dict["C"].set_ylim((-0.03,1.03))
    ax_dict["C"].set_xlabel(r'PINUP method')
    ax_dict["C"].set_ylabel(r'$R^2$', labelpad=5)
    ax_dict["C"].set_xticks(range(1, len(METHODS)+2))
    ax_dict["C"].set_xticklabels(METHODS + [fname])
    ax_dict["C"].spines['right'].set_visible(False)
    ax_dict["C"].spines['top'].set_visible(False)
    ax_dict["C"].grid(visible=True, axis='y', linewidth=0.4)

    for i, (noise, marker) in enumerate(zip(NOISE, ['o', '^', 's'])):
        for k, (method, col) in enumerate(zip(METHODS + [fname], ['green', 'blue', 'orange', 'black'])):
            print(np.std(RHOS[i,:,k]))
            ax_dict["C"].errorbar(k + 1 + (i -1)*0.3, np.mean(RHOS[i,:,k], axis=0), yerr=np.std(RHOS[i,:,k], axis=0),
                                label=method, 
                                c=PALETTE[col], linewidth=1, marker=marker, markersize=6)
    if legend:
        legend1 = ax_dict["A"].errorbar([],[],color='black', marker='o', linestyle='-', label='Noise-free')
        legend2 = ax_dict["A"].errorbar([],[],color='black', marker='^', linestyle='-', label='SNR = 20 dB')
        legend3 = ax_dict["A"].errorbar([],[],color='black', marker='s', linestyle='-', label='SNR = 0 dB')
        #plt.legend(handles=[legend1, legend2, legend3], loc=(1.05, 0.5), frameon=False)
        ax_dict["C"].legend(handles=[legend1, legend2, legend3], loc='center left', frameon=False)

    fig.text(0.02, 1.05, r'$\textbf{s}$'.replace('s', text[0]), fontsize=18)
    fig.text(0.02, 0.35, r'$\textbf{s}$'.replace('s', text[1]), fontsize=18)
    fig.text(0.42, 1.05, r'$\textbf{s}$'.replace('s', text[2]), fontsize=18)

    return fig