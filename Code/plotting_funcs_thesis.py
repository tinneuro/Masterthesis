import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from matplotlib import cycler
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as colors
# importing own scripts
import statistics as stats_UD
import model_2D


# colors from ggplot
my_colors = cycler('color',
                ['#E24A33','black','#348ABD','#988ED5','#777777','#FBC15E','#8EBA42','#FFB5B8'])

def heatmap_with_bifurcation(matrix,x_grid,y_grid,line1_x,line1_y,line2_x,line2_y,title,xlabel,ylabel,colorbarlabel,round_int = 1,round_int2 = 0,max_float = 100,norm = None,vmin=None, vmax=None,equal_dur = False,saveplot = False,save_path = None):
    """This is for the 2D model to plot a heatmap of the different statistics and the corresponding bifurcationlines.
        round_int: to how many digits the numbers on the squares are rounded
        Set max_float value up to which the numbers are rounded to round_int, when the entries are bigger than max_float (default 100) then we use round_int2.
        equal_dur plots the I_1/2 line the diagram
        matrix is the matrix with statistics
        x_grid is the array of values on the xaxis
        y_grid is the array of values on the yaxis
        line1_x,line2_x,line1_y,line2_y are the bifurcation lines"""
    params = {'figure.figsize': (20,20),
          'lines.linewidth': 4,
          'legend.fontsize': 20,
         'axes.labelsize': 40,
         'axes.titlesize':45,
         'xtick.labelsize':35,
         'ytick.labelsize':35,
         'xtick.major.size': 10,
          'xtick.major.width' : 2,
          'xtick.minor.size' :5,
          'xtick.minor.width' : 1,
         'ytick.major.size': 10,
          'ytick.major.width' : 2,
          'ytick.minor.size' :5,
          'ytick.minor.width' : 1,
         'figure.constrained_layout.use': True}
    plt.rcParams.update(params)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    textcolors=("black", "white")

    x, y = np.meshgrid(x_grid,y_grid)
    dist_x = np.round(x_grid[1]-x_grid[0],2)/2 # calculating half of the distance between to gridpoints to get the ticks into the middle
    dist_y = np.round(y_grid[0]-y_grid[1],2)/2

    # plotting
    ax.plot(line1_x,line1_y,color = "black")#plotting bifurcation lines
    ax.plot(line2_x,line2_y,color = "black",linestyle = "--")
    if equal_dur:
        ax.plot(x_grid,-0.2*x_grid+2.4,color = "red",linewidth = 4)
    # extent set such that the ticks are in the middle of the squares
    heatmap = ax.imshow(matrix,extent=[x.min()-dist_x, x.max()+dist_x, y.min()-dist_y, y.max()+dist_y], origin = "upper",cmap = "plasma",aspect = 4,norm = norm,vmin = vmin,vmax = vmax)
    cbar = fig.colorbar(heatmap, ax=ax,shrink = 0.75)#fraction can resize the colorbar
    cbar.set_label(colorbarlabel,fontsize = 40)
    ax.set_xticks(x_grid)
    ax.set_yticks(y_grid)
    ax.set_xticklabels(np.round(x_grid,2), rotation=90) # rotate the xticks such that still readable for more comma vals
    ax.set_yticklabels(np.round(y_grid,2))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Loop over data dimensions and create text annotations.
    for (idxi,i) in enumerate(y_grid):
        for (idxj,j) in enumerate(x_grid):
            if not np.isnan(matrix[idxi, idxj]):# do not want to display the nan values
                if np.round(matrix[idxi, idxj],round_int) < max_float:
                    text = ax.text(j, i, np.round(matrix[idxi, idxj],round_int),
                               ha="center", va="center", color=textcolors[int(heatmap.norm(matrix[idxi, idxj]) < 0.5)],size = 18)# might need to reset threshold for other graphics
                else:
                    if round_int2 == 0:
                        text = ax.text(j, i, int(np.round(matrix[idxi, idxj],round_int2)),
                               ha="center", va="center", color=textcolors[int(heatmap.norm(matrix[idxi, idxj]) < 0.5)],size = 18)
                    else:
                        text = ax.text(j, i, np.round(matrix[idxi, idxj],round_int2),
                               ha="center", va="center", color=textcolors[int(heatmap.norm(matrix[idxi, idxj]) < 0.5)],size = 18)


    # possibility to save the plot
    if saveplot:
        plt.savefig(save_path,dpi=200)
    plt.show()

def heatmap_with_bifurcation_compare(matrix_mult,matrix_add,x_grid,y_grid,line1_x,line1_y,line2_x,line2_y,title_mult,title_add,xlabel,ylabel,colorbarlabel,round_int = 1,round_int2 = 0,max_float = 100,norm = None,vmin=None, vmax=None,equal_dur = False,saveplot = False,save_path = None):
    """Same as heatmap_with_bifurcation but it plots two of these diagrams next to each other for comparison"""
    params = {'figure.figsize': (20,20),
          'lines.linewidth': 4,
          'legend.fontsize': 20,
         'axes.labelsize': 40,
         'axes.titlesize':45,
         'xtick.labelsize':35,
         'ytick.labelsize':35,
         'xtick.major.size': 10,
          'xtick.major.width' : 2,
          'xtick.minor.size' :5,
          'xtick.minor.width' : 1,
         'ytick.major.size': 10,
          'ytick.major.width' : 2,
          'ytick.minor.size' :5,
          'ytick.minor.width' : 1}
    plt.rcParams.update(params)
    fig,(ax1,ax2) = plt.subplots(1,2,figsize = (40,40),constrained_layout = True)
    textcolors=("black", "white")

    x, y = np.meshgrid(x_grid,y_grid)
    dist_x = np.round(x_grid[1]-x_grid[0],2)/2 # calculating half of the distance between to gridpoints to get the ticks into the middle
    dist_y = np.round(y_grid[0]-y_grid[1],2)/2
    # plotting mult results
    ax1.plot(line1_x,line1_y,color = "black")#plotting bifurcation lines
    ax1.plot(line2_x,line2_y,color = "black",linestyle = "--")
    if equal_dur:
        ax1.plot(x_grid,-0.2*x_grid+2.4,color = "red",linewidth = 4)
    # extent set such that the ticks are in the middle of the squares
    heatmap = ax1.imshow(matrix_mult,extent=[x.min()-dist_x, x.max()+dist_x, y.min()-dist_y, y.max()+dist_y], origin = "upper",cmap = "plasma",aspect = 4,norm = norm,vmin = vmin,vmax = vmax)
    cbar = fig.colorbar(heatmap, ax=(ax1,ax2),shrink = 0.4,aspect = 8)#fraction can resize the colorbar
    cbar.set_label(colorbarlabel,fontsize = 40)
    ax1.set_xticks(x_grid)
    ax1.set_yticks(y_grid)
    ax1.set_xticklabels(np.round(x_grid,2), rotation=90) # rotate the xticks such that still readable for more comma vals
    ax1.set_yticklabels(np.round(y_grid,2))
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title_mult)

    # Loop over data dimensions and create text annotations.
    for (idxi,i) in enumerate(y_grid):
        for (idxj,j) in enumerate(x_grid):
            if not np.isnan(matrix_mult[idxi, idxj]):# do not want to display the nan values
                if np.round(matrix_mult[idxi, idxj],round_int) < max_float:
                    text = ax1.text(j, i, np.round(matrix_mult[idxi, idxj],round_int),
                               ha="center", va="center", color=textcolors[int(heatmap.norm(matrix_mult[idxi, idxj]) < 0.5)],size = 18)# might need to reset threshold for other graphics
                else:
                    if round_int2 == 0:
                        text = ax1.text(j, i, int(np.round(matrix_mult[idxi, idxj],round_int2)),
                               ha="center", va="center", color=textcolors[int(heatmap.norm(matrix_mult[idxi, idxj]) < 0.5)],size = 18)
                    else:
                        text = ax1.text(j, i, np.round(matrix_mult[idxi, idxj],round_int2),
                               ha="center", va="center", color=textcolors[int(heatmap.norm(matrix_mult[idxi, idxj]) < 0.5)],size = 18)


    # plotting add results
    ax2.plot(line1_x,line1_y,color = "black")#plotting bifurcation lines
    ax2.plot(line2_x,line2_y,color = "black",linestyle = "--")
    if equal_dur:
        ax2.plot(x_grid,-0.2*x_grid+2.4,color = "red",linewidth = 4)
    # extent set such that the ticks are in the middle of the squares
    heatmap = ax2.imshow(matrix_add,extent=[x.min()-dist_x, x.max()+dist_x, y.min()-dist_y, y.max()+dist_y], origin = "upper",cmap = "plasma",aspect = 4,norm = norm,vmin = vmin,vmax = vmax)
    ax2.set_xticks(x_grid)
    ax2.set_yticks(y_grid)
    ax2.set_xticklabels(np.round(x_grid,2), rotation=90) # rotate the xticks such that still readable for more comma vals
    ax2.set_yticklabels(np.round(y_grid,2))
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax2.set_title(title_add)

    # Loop over data dimensions and create text annotations.
    for (idxi,i) in enumerate(y_grid):
        for (idxj,j) in enumerate(x_grid):
            if not np.isnan(matrix_add[idxi, idxj]):# do not want to display the nan values
                if np.round(matrix_add[idxi, idxj],round_int) < max_float:
                    text = ax2.text(j, i, np.round(matrix_add[idxi, idxj],round_int),
                               ha="center", va="center", color=textcolors[int(heatmap.norm(matrix_add[idxi, idxj]) < 0.5)],size = 18)# might need to reset threshold for other graphics
                else:
                    if round_int2 == 0:
                        text = ax2.text(j, i, int(np.round(matrix_add[idxi, idxj],round_int2)),
                               ha="center", va="center", color=textcolors[int(heatmap.norm(matrix_add[idxi, idxj]) < 0.5)],size = 18)
                    else:
                        text = ax2.text(j, i, np.round(matrix_add[idxi, idxj],round_int2),
                               ha="center", va="center", color=textcolors[int(heatmap.norm(matrix_add[idxi, idxj]) < 0.5)],size = 18)


    # possibility to save the plot
    if saveplot:
        plt.savefig(save_path,dpi=200)
    plt.show()

def heatmap_with_bifurcation_compare_four(matrices,x_grid,y_grid,line1_x,line1_y,line2_x,line2_y,titles,xlabel,ylabel,colorbarlabel,round_int = 1,round_int2 = 0,max_float = 100,norm = None,vmin=None, vmax=None,equal_dur = False,saveplot = False,save_path = None):
    """same functionalities as heatmap_with_bifurcation but we compare 4 grids with each other
    grid with four sections, Mult, mean addconst, low add const, high add const"""
    params = {'figure.figsize': (20,20),
          'lines.linewidth': 4,
          'legend.fontsize': 20,
         'axes.labelsize': 40,
         'axes.titlesize':45,
         'xtick.labelsize':35,
         'ytick.labelsize':35,
         'xtick.major.size': 10,
          'xtick.major.width' : 2,
          'xtick.minor.size' :5,
          'xtick.minor.width' : 1,
         'ytick.major.size': 10,
          'ytick.major.width' : 2,
          'ytick.minor.size' :5,
          'ytick.minor.width' : 1}
    plt.rcParams.update(params)

    #matrices = [matrix_mult,matrix_add1,matrix_add2,matrix_add3]
    fig, axs = plt.subplots(nrows=2, ncols=2,figsize = (40,40),constrained_layout = True)
    textcolors=("black", "white")

    x, y = np.meshgrid(x_grid,y_grid)
    dist_x = np.round(x_grid[1]-x_grid[0],2)/2 # calculating half of the distance between to gridpoints to get the ticks into the middle
    dist_y = np.round(y_grid[0]-y_grid[1],2)/2

    for i,ax in enumerate(axs.reshape(-1)):
    # plotting mult results
        ax.plot(line1_x,line1_y,color = "black")#plotting bifurcation lines
        ax.plot(line2_x,line2_y,color = "black",linestyle = "--")
        if equal_dur:
            ax.plot(x_grid,-0.2*x_grid+2.4,color = "red",linewidth = 4)
    # extent set such that the ticks are in the middle of the squares
        heatmap = ax.imshow(matrices[i],extent=[x.min()-dist_x, x.max()+dist_x, y.min()-dist_y, y.max()+dist_y], origin = "upper",cmap = "plasma",aspect = 4,norm = norm,vmin = vmin,vmax = vmax)
        ax.set_xticks(x_grid)
        ax.set_yticks(y_grid)
        ax.set_xticklabels(np.round(x_grid,2), rotation=90) # rotate the xticks such that still readable for more comma vals
        ax.set_yticklabels(np.round(y_grid,2))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(titles[i])

        # Loop over data dimensions and create text annotations.
        matrix = matrices[i]
        for (idxi,i) in enumerate(y_grid):
            for (idxj,j) in enumerate(x_grid):
                if not np.isnan(matrix[idxi, idxj]):# do not want to display the nan values
                    if np.round(matrix[idxi, idxj],round_int) < max_float:
                        text = ax.text(j, i, np.round(matrix[idxi, idxj],round_int),
                                   ha="center", va="center", color=textcolors[int(heatmap.norm(matrix[idxi, idxj]) < 0.5)],size = 18)# might need to reset threshold for other graphics
                    else:
                        if round_int2 == 0:
                            text = ax.text(j, i, int(np.round(matrix[idxi, idxj],round_int2)),
                                   ha="center", va="center", color=textcolors[int(heatmap.norm(matrix[idxi, idxj]) < 0.5)],size = 18)
                        else:
                            text = ax.text(j, i, np.round(matrix[idxi, idxj],round_int2),
                                   ha="center", va="center", color=textcolors[int(heatmap.norm(matrix[idxi, idxj]) < 0.5)],size = 18)

    cbar = fig.colorbar(heatmap, ax=axs,shrink = 0.4,aspect = 8)#fraction can resize the colorbar
    cbar.set_label(colorbarlabel,fontsize = 40)

    # possibility to save the plot
    if saveplot:
        plt.savefig(save_path,dpi=200)
    plt.show()

def heatmap_diagonals(matrix,matrix_notnan_add,x_grid,y_grid,line1_x,line1_y,line2_x,line2_y,title,xlabel,ylabel,colorbarlabel,round_int = 1,round_int2 = 0,max_float = 100,norm = None,vmin=None, vmax=None,equal_dur = False,saveplot = False,save_path = None):
    """Heatmap for the plot showing the diagonals and sub or super diagonals
    matrix_notnan_add is the add matrix only used to set the color range."""
    params = {'figure.figsize': (20,20),
          'lines.linewidth': 4,
          'legend.fontsize': 20,
         'axes.labelsize': 40,
         'axes.titlesize':45,
         'xtick.labelsize':35,
         'ytick.labelsize':35,
         'xtick.major.size': 10,
          'xtick.major.width' : 2,
          'xtick.minor.size' :5,
          'xtick.minor.width' : 1,
         'ytick.major.size': 10,
          'ytick.major.width' : 2,
          'ytick.minor.size' :5,
          'ytick.minor.width' : 1,
         'figure.constrained_layout.use': True}
    plt.rcParams.update(params)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    textcolors=("black", "white")
    bounds = range(int(min(matrix_notnan_add)),int(max(matrix_notnan_add))+1,1)
    cmap_20 = plt.cm.get_cmap("tab20")
    norm = colors.BoundaryNorm(bounds, cmap_20.N)

    x, y = np.meshgrid(x_grid,y_grid)
    dist_x = np.round(x_grid[1]-x_grid[0],2)/2 # calculating half of the distance between to gridpoints to get the ticks into the middle
    dist_y = np.round(y_grid[0]-y_grid[1],2)/2

    # plotting
    ax.plot(line1_x,line1_y,color = "black")#plotting bifurcation lines
    ax.plot(line2_x,line2_y,color = "black",linestyle = "--")
    if equal_dur:
        ax.plot(x_grid,-0.2*x_grid+2.4,color = "red",linewidth = 4)
    # extent set such that the ticks are in the middle of the squares
    heatmap = ax.imshow(matrix,extent=[x.min()-dist_x, x.max()+dist_x, y.min()-dist_y, y.max()+dist_y], origin = "upper",cmap = "tab20",aspect = 4,norm = norm,vmin = vmin,vmax = vmax)
    cbar = fig.colorbar(heatmap, ax=ax,shrink = 0.75,ticks=np.array(np.arange(-10,12,1))+0.5)#fraction can resize the colorbar
    cbar.set_label(colorbarlabel,fontsize = 40)
    cbar.set_ticklabels(np.array(np.arange(-5,10,1)))
    ax.set_xticks(x_grid)
    ax.set_yticks(y_grid)
    ax.set_xticklabels(np.round(x_grid,2), rotation=90) # rotate the xticks such that still readable for more comma vals
    ax.set_yticklabels(np.round(y_grid,2))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Loop over data dimensions and create text annotations.
    for (idxi,i) in enumerate(y_grid):
        for (idxj,j) in enumerate(x_grid):
            if not np.isnan(matrix[idxi, idxj]):# do not want to display the nan values
                    text = ax.text(j, i, int(np.round(matrix[idxi, idxj],round_int2)),
                            ha="center", va="center", color=textcolors[int(heatmap.norm(matrix[idxi, idxj]) < 0.5)],size = 18)

    # possibility to save the plot
    if saveplot:
        plt.savefig(save_path,dpi=200)
    plt.show()

def plot_dynamics_2D(sim_time,rate,adaptation,start_time = None,end_time = None,save_path = None,saveplot = False):
    """Plotting 2D dynamics firing rate and adaptation"""
    plt.style.use('default')
    params = {'figure.figsize': (28,14),
            'axes.prop_cycle': my_colors,
              'lines.linewidth': 5,
              'legend.fontsize': 40,
             'axes.labelsize': 50,
             'axes.titlesize':60,
             'xtick.labelsize':45,
             'ytick.labelsize':45,
              'xtick.major.size': 16,
               'xtick.major.width' : 2,
               'xtick.minor.size' :10,
               'xtick.minor.width' : 2,
              'ytick.major.size': 16,
               'ytick.major.width' : 2,
               'ytick.minor.size' :10,
               'ytick.minor.width' : 2,
             'figure.constrained_layout.use': False}
    plt.rcParams.update(params)

    if start_time != None and end_time != None:
        time_step = sim_time[1]-sim_time[0]
        start_time = int(start_time/time_step)
        end_time = int(end_time/time_step)

    sim_time_cut = sim_time[start_time:end_time]
    rate_cut = rate[start_time:end_time]
    adapt_cut = adaptation[start_time:end_time]

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))# restrict decimal to 2
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))# restrict decimal to 2

    ax1.plot(sim_time_cut,rate_cut)
    ax1.set_xlabel("s")
    ax1.set_ylabel("Hz")
    ax1.set_yticks(np.arange(0,max(rate_cut)+2.5,step=2.5))
    ax1.set_title("Firing rate: r")

    ax2.plot(sim_time_cut,adapt_cut,color = "black")
    ax2.set_yticks(np.arange(round(min(adapt_cut),2)-0.5,round(max(adapt_cut),2)+0.5,step=0.5))
    ax2.set_title("Adaptation: a")
    ax2.set_xlabel("s")
    ax2.set_ylabel("mV")
    plt.tight_layout()

    # possibility to save the plot
    if saveplot:
        plt.savefig(save_path,dpi=200)

    plt.show()

def plot_exc_only_2D(sim_time,rate,start_time = None,end_time = None,save_path = None,saveplot = False):
    """Plotting 2D dynamics firing rate only"""
    plt.style.use('default')
    params = {'figure.figsize': (28,7),
            'axes.prop_cycle': my_colors,
              'lines.linewidth': 5,
              'legend.fontsize': 40,
             'axes.labelsize': 50,
             'axes.titlesize':60,
             'xtick.labelsize':45,
             'ytick.labelsize':45,
              'xtick.major.size': 16,
               'xtick.major.width' : 2,
               'xtick.minor.size' :10,
               'xtick.minor.width' : 2,
              'ytick.major.size': 16,
               'ytick.major.width' : 2,
               'ytick.minor.size' :10,
               'ytick.minor.width' : 2,
             'figure.constrained_layout.use': False}
    plt.rcParams.update(params)

    if start_time != None and end_time != None:
        time_step = sim_time[1]-sim_time[0]
        start_time = int(start_time/time_step)
        end_time = int(end_time/time_step)

    sim_time_cut = sim_time[start_time:end_time]
    rate_cut = rate[start_time:end_time]

    fig, ax1 = plt.subplots(1)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))# restrict decimal to 2

    ax1.plot(sim_time_cut,rate_cut)
    ax1.set_xlabel("s")
    ax1.set_ylabel("Hz")
    ax1.set_yticks(np.arange(0,max(rate_cut)+2.5,step=2.5))
    plt.tight_layout()

    # possibility to save the plot
    if saveplot:
        plt.savefig(save_path,dpi=200)

    plt.show()


def plot_dynamics_2D_marker(sim_time,rate,adaptation,start_time = None,end_time = None,start_traj = None,end_traj = None,save_path = None,saveplot = False):
    """Plotting 2D dynamics firing rate and adaptation with markers to mark trajectory"""
    plt.style.use('default')
    params = {'figure.figsize': (28,14),
            'axes.prop_cycle': my_colors,
              'lines.linewidth': 5,
              'legend.fontsize': 40,
             'axes.labelsize': 50,
             'axes.titlesize':60,
             'xtick.labelsize':45,
             'ytick.labelsize':45,
              'xtick.major.size': 16,
               'xtick.major.width' : 2,
               'xtick.minor.size' :10,
               'xtick.minor.width' : 2,
              'ytick.major.size': 16,
               'ytick.major.width' : 2,
               'ytick.minor.size' :10,
               'ytick.minor.width' : 2,
             'figure.constrained_layout.use': False}
    plt.rcParams.update(params)

    if start_time != None and end_time != None:
        time_step = sim_time[1]-sim_time[0]
        start_time = int(start_time/time_step)
        end_time = int(end_time/time_step)

    if start_traj != None and end_traj != None:
        time_step = sim_time[1]-sim_time[0]
        start_traj = int(start_traj/time_step)
        end_traj = int(end_traj/time_step)

    sim_time_cut = sim_time[start_time:end_time]
    rate_cut = rate[start_time:end_time]
    adapt_cut = adaptation[start_time:end_time]

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))# restrict decimal to 2
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))# restrict decimal to 2

    ax1.plot(sim_time_cut,rate_cut)
    ax1.plot(sim_time[start_traj],rate[start_traj],marker = "^",mew = 5,ms = 20,color = 'black')
    ax1.plot(sim_time[end_traj],rate[end_traj],marker = "v",mew = 5,ms = 20,color = 'black')
    ax1.set_xlabel("s")
    ax1.set_ylabel("Hz")
    ax1.set_yticks(np.arange(0,max(rate_cut)+2.5,step=2.5))
    ax1.set_title("Firing rate: r")

    ax2.plot(sim_time_cut,adapt_cut,color = "black")
    ax2.plot(sim_time[start_traj],adaptation[start_traj],marker = "^",mew = 5,ms = 20,color = 'black')
    ax2.plot(sim_time[end_traj],adaptation[end_traj],marker = "v",mew = 5,ms = 20,color = 'black')
    ax2.set_yticks(np.arange(round(min(adapt_cut),2)-0.5,round(max(adapt_cut),2)+0.5,step=0.5))
    ax2.set_title("Adaptation: a")
    ax2.set_xlabel("s")
    ax2.set_ylabel("mV")
    plt.tight_layout()

    # possibility to save the plot
    if saveplot:
        plt.savefig(save_path,dpi=200)

    plt.show()

def mov_avg_plot(sim_time,rate,mov_avg,window_avg,dt,start_time = None,end_time = None,saveplot = False,save_path = None):
    """Plotting firing rate and moving average above"""
    plt.style.use('default')

    params = {'figure.figsize': (28,7),
            'axes.prop_cycle': my_colors,
              'lines.linewidth': 5,
              'legend.fontsize': 40,
             'axes.labelsize': 50,
             'axes.titlesize':60,
             'xtick.labelsize':45,
             'ytick.labelsize':45,
              'xtick.major.size': 16,
               'xtick.major.width' : 2,
               'xtick.minor.size' :10,
               'xtick.minor.width' : 2,
              'ytick.major.size': 16,
               'ytick.major.width' : 2,
               'ytick.minor.size' :10,
               'ytick.minor.width' : 2,
             'figure.constrained_layout.use': False}
    plt.rcParams.update(params)

    if start_time != None and end_time != None:
        start_time = int(start_time/dt)
        end_time = int(end_time/dt)

    sim_time_cut = sim_time[start_time:end_time]
    rate_cut = rate[start_time:end_time]
    mov_avg_cut = mov_avg[start_time:end_time]

    plt.plot(sim_time_cut,rate_cut,label = "Firing Rate excitatory")
    plt.plot(sim_time_cut,mov_avg_cut,linewidth = 4,label = f"Moving average with window {window_avg*dt}s")
    #plt.yticks(np.arange(0,max(rate_cut),step=5))

    plt.xlabel("s")
    plt.ylabel("Hz")
    #plt.ylim(0,15)
    plt.yticks([0,5,10,15])
    #plt.legend(fontsize = 25,loc = 'center left',bbox_to_anchor=(1, 0.5))

    if saveplot:
        plt.savefig(save_path,dpi=200)
    plt.show()

def mov_avg_crossings_plot(sim_time,rate,mov_avg,crossings,window_avg,dt,threshold_up,start_time = None,end_time = None,show_legend = False,saveplot = False,save_path = None):
    """Moving average plot with crossings using only one threshold (threshold_up)."""

    plt.style.use('default')
    params = {'figure.figsize': (28,7),
            'axes.prop_cycle': my_colors,
              'lines.linewidth': 5,
              'legend.fontsize': 40,
             'axes.labelsize': 50,
             'axes.titlesize':60,
             'xtick.labelsize':45,
             'ytick.labelsize':45,
              'xtick.major.size': 16,
               'xtick.major.width' : 2,
               'xtick.minor.size' :10,
               'xtick.minor.width' : 2,
              'ytick.major.size': 16,
               'ytick.major.width' : 2,
               'ytick.minor.size' :10,
               'ytick.minor.width' : 2,
             'figure.constrained_layout.use': False}
    plt.rcParams.update(params)

    if start_time != None and end_time != None:
        start_time = int(start_time/dt)
        end_time = int(end_time/dt)

    sim_time_cut = sim_time[start_time:end_time]
    rate_cut = rate[start_time:end_time]
    mov_avg_cut = mov_avg[start_time:end_time]
    # find the crossings corresponding to the zoom in time window start_time:end_time
    crossings_cut = crossings[np.logical_and(crossings>=start_time, crossings<=end_time)]

    plt.plot(sim_time_cut,rate_cut,label = "Firing Rate excitatory")
    plt.plot(sim_time_cut,mov_avg_cut,label = f"Moving average with window {window_avg*dt}s")
    # for this we still need the whole sim time because we use the cutted indices of the crossings
    plt.plot(sim_time[crossings_cut],mov_avg[crossings_cut],marker = "x",color = "black",mew = 4,markersize=18,label = "Crossings")
    plt.plot(sim_time_cut,threshold_up*np.ones(len(sim_time_cut)),color = "black",label = f"Threshold of {threshold_up} Hz")
    plt.xlabel("s")
    plt.ylabel("Hz")
    if show_legend:
        plt.legend(fontsize = 25,loc = "upper right")

    # possibility to save the plot
    if saveplot:
        plt.savefig(save_path,dpi=200)
    plt.show()

def mov_avg_crossings_2thr_plot(sim_time,rate,mov_avg,crossings,window_avg,threshold_DU,threshold_UD,dt,start_time = None,end_time = None,show_legend = False,saveplot = False,save_path = None):
    """Plotting firing rate, moving average and crossings, thresholds for the two thresholds classification."""
    plt.style.use('default')
    params = {'figure.figsize': (28,7),
            'axes.prop_cycle': my_colors,
              'lines.linewidth': 5,
              'legend.fontsize': 40,
             'axes.labelsize': 50,
             'axes.titlesize':60,
             'xtick.labelsize':45,
             'ytick.labelsize':45,
              'xtick.major.size': 16,
               'xtick.major.width' : 2,
               'xtick.minor.size' :10,
               'xtick.minor.width' : 2,
              'ytick.major.size': 16,
               'ytick.major.width' : 2,
               'ytick.minor.size' :10,
               'ytick.minor.width' : 2,
             'figure.constrained_layout.use': False}
    plt.rcParams.update(params)


    if start_time != None and end_time != None:
        start_time = int(start_time/dt)
        end_time = int(end_time/dt)

    sim_time_cut = sim_time[start_time:end_time]
    rate_cut = rate[start_time:end_time]
    mov_avg_cut = mov_avg[start_time:end_time]
    # find the crossings corresponding to the zoom in time window start_time:end_time
    crossings_cut = crossings[np.logical_and(crossings>=start_time, crossings<=end_time)]

    plt.plot(sim_time_cut,rate_cut,zorder=1,label = "Firing rate")
    plt.plot(sim_time_cut,mov_avg_cut,linewidth = 4,zorder=2,label = f"Moving average")

    plt.plot(sim_time_cut,threshold_DU*np.ones(len(sim_time_cut)),color = "black",zorder=3,label = f"horizontal: Thresholds")
    plt.plot(sim_time_cut,threshold_UD*np.ones(len(sim_time_cut)),color = "black",zorder=4)
    # for this we still need the whole sim time because we use the cutted indices of the crossings
    plt.scatter(sim_time[crossings_cut],mov_avg[crossings_cut],marker = "x",s = 360,c = "black",linewidth=4,zorder=5,label = "Crossings")
    plt.xlabel("s")
    plt.ylabel("Hz")

    if show_legend:
        plt.legend(fontsize = 50,loc = "upper right",bbox_to_anchor=(1.6, 0.5))


    # possibility to save the plot
    if saveplot:
        plt.savefig(save_path,dpi=200)
    plt.show()
