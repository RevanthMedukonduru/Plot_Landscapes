import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import seaborn as sns
import torch
from scipy.spatial import distance
from scipy.interpolate import griddata
import numpy as np
import torch

matplotlib.rc('text', usetex=True)
latex_preamble = '\\usepackage{sansmath} \\sansmath'
matplotlib.rcParams['text.latex.preamble'] = latex_preamble
matplotlib.rc('font', **{'family':'sans-serif','sans-serif':['DejaVu Sans']})
matplotlib.rc('xtick.major', pad=12)
matplotlib.rc('ytick.major', pad=12)
matplotlib.rc('grid', linewidth=0.8)
sns.set_style('whitegrid')

import argparse

# argparse to load params
args = argparse.ArgumentParser(description='Plot the landscape of the given surface')
args.add_argument('--path', type=str, default="", required=True, help='Path to the folder where npz file is available')
args.add_argument('--whatToPlot', type=str, required=True, help='What do I want to plot? with respect to either tr_loss, tr_acc, tr_nll, tr_err, te_loss, te_acc, te_nll, te_err or all option to plot all', choices=['tr_loss', 'tr_err', 'te_loss', 'te_err', 'all'])
args.add_argument('--contour', type=bool, default=False, help='Do you want to plot contour? means with values on the contour lines')
args.add_argument('--saveAs', type=str, required=True, help='Do you want to save the plot as svg vector file or png format?', choices=['svg', 'png', 'both'])
args.add_argument('--connectOptimas', type=bool, default=False, help='Do you want to connect the optimas with a line?')
args = args.parse_args()

path = args.path
what_do_i_want_to_plot = args.whatToPlot
print("Args: ", args)

data = dict(np.load(path+"plane.npz"))
print("KEYS AVAIABLE IN GIVEN NPZ FILE IS:", data.keys())

def Plot_Surfaces(data, feature):
    # get the grid values
    grid = data['grid'][0]
    values = data[feature][0]
    # print(grid.shape, values.shape) # (G, G, 2) (G, G) G-> No of points we choose default 40

    # Prepare the plot
    plt.figure(figsize=(10, 8))
    plt.grid(False)
    plt.contourf(grid[:, :, 0], grid[:, :, 1], values, levels=80, cmap='jet_r', alpha=0.8)
    plt.colorbar(label=feature)

    if args.contour:
        contours = plt.contour(grid[:, :, 0], grid[:, :, 1], values, levels=30, colors='black', linewidths=0.2)
        plt.clabel(contours, inline=True, fontsize=9)

    # get the bend_coordinates
    bend_coordinates = data['bend_coordinates']

    # Colors for different sets of optimas
    colors_1 = ['red', 'blue', 'green']  

    for i, bend_coord in enumerate(bend_coordinates): 
        plt.scatter(bend_coord[0], bend_coord[1], c=colors_1[i], s=100, label=f'Optima {i+1}', edgecolors='black', zorder=1)

    # data type of args.connectOptimas is bool
    print(args.connectOptimas, "Connect optimas", type(args.connectOptimas))
    if args.connectOptimas:    
        plt.plot(bend_coordinates[:, 0], bend_coordinates[:, 1], c='k', linestyle='--', dashes=(3, 4), linewidth=3, zorder=2)
        plt.plot(bend_coordinates[[0, 2], 0], bend_coordinates[[0, 2], 1], c='k', linestyle='--', dashes=(3, 4), linewidth=3, zorder=2)
        
    plt.xlabel('Dimension 1 (Moving across the plane from Optima 1 to Optima 3)')
    plt.ylabel('Dimension 2 (Moving across the plane from Optima 1 to Optima 2)')
    plt.title(f'Interpolation between Optimas - Loss Surface of {feature}')
    plt.legend()
    
    if args.saveAs!="both":
        plt.savefig(f"{path}/{feature}_surface.{args.saveAs}", format=str(args.saveAs))
    else:
        plt.savefig(f"{path}/{feature}_surface.svg", format='svg')
        plt.savefig(f"{path}/{feature}_surface.png", format='png')


if __name__ == '__main__':
    if what_do_i_want_to_plot == 'all':
        for feature in ['tr_loss', 'tr_err', 'te_loss', 'te_err']:
            Plot_Surfaces(data, feature)
            print(f"Plotting done for {feature} surface")
    else:
        Plot_Surfaces(data, what_do_i_want_to_plot)
        
    print("Plotting for the given features is finished!")
