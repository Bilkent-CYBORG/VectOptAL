import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
from matplotlib.patches import Polygon
from matplotlib.pyplot import cm
import os

def plot_func_list(list, range1, range2, title1, title2, no_points=None, h = None):
    if h is None:
        x1 = np.linspace(range1[0], range1[1], no_points)
        x2 = np.linspace(range2[0], range2[1], no_points)
    else:
        x1 = np.linspace(range1[0], range1[1], h+1)
        x2 = np.linspace(range2[0], range2[1], h+1)
    px1, px2 = np.meshgrid(x1, x2)

    fig = plt.figure(figsize=plt.figaspect(0.4))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    func_val1 = list[0](np.vstack((px1.flatten(), px2.flatten())).T)
    ax1.scatter3D(px1, px2,  func_val1, c=func_val1, cmap='viridis')
    ax1.set_xlabel('$X_1$')
    ax1.set_ylabel('$X_2$')
    ax1.set_title(title1)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    func_val2 = list[1](np.vstack((px1.flatten(), px2.flatten())).T)
    ax2.scatter3D(px1, px2, func_val2, c=func_val2, cmap='viridis')
    ax2.set_xlabel('$X_1$')
    ax2.set_ylabel('$X_2$')
    ax2.set_title(title2)
    plt.show()

    return func_val1, func_val2


def plot_func_list_1d(list, range1, title1, title2, no_points=None, h = None):

    px1 = np.linspace(range1[0], range1[1], no_points).reshape(no_points, 1)


    fig = plt.figure(figsize=plt.figaspect(0.4))
    ax1 = fig.add_subplot(1, 2, 1)
    func_val1 = list[0](px1)
    ax1.scatter(px1, func_val1, cmap='viridis')
    ax1.set_xlabel('$X_1$')
    ax1.set_ylabel('')
    ax1.set_title(title1)

    ax2 = fig.add_subplot(1, 2, 2)
    func_val2 = list[1](np.vstack(px1))
    ax2.scatter(px1, func_val2, cmap='viridis')
    ax2.set_xlabel('$X_1$')
    ax2.set_ylabel('')
    ax2.set_title(title2)
    plt.show()

    return func_val1, func_val2


def plot_pareto_front(func_val1 = None, func_val2 = None, mask = None, y1 = None, y2 = None, title=None, plotfront = False,xlabel=None,ylabel=None,save_filename=None):
    fig = plt.figure(figsize=(8, 5))
    # plt.rcParams["figure.dpi"] = 400
    ax = plt.axes()
    if func_val1 is not None:

        ax.scatter(func_val1[mask ==False], func_val2[mask == False], color='gray', alpha=0.5)
        ax.scatter(func_val1[mask], func_val2[mask], c='tab:green',label="Gerçek Pareto", alpha=0.7)
        """
        func_val1sorted = np.sort(func_val1[mask])
        func_val2sorted = np.sort(func_val2[mask])[::-1]

         for i in range (func_val1[mask].shape[0] - 1):
            x_values = [func_val1sorted [i], func_val1sorted[i]]
            y_values = [func_val2sorted[i], func_val2sorted[i+1]]
            plt.plot(x_values, y_values, color='darkseagreen')
            x_values = [func_val1sorted[i], func_val1sorted[i+1]]
            y_values = [func_val2sorted[i+1], func_val2sorted[i + 1]]
            plt.plot(x_values, y_values, color='darkseagreen') """



    if y1 is not None:
        ax.scatter(y1, y2,s=16,marker="x", c='tab:red',label="Tahmini Pareto", alpha=0.9)
        """ func_val1sorted = np.sort(y1)
        func_val2sorted = np.sort(y2)[::-1]
        
        if plotfront:
            for i in range(y1.shape[0] - 1):
                x_values = [func_val1sorted[i], func_val1sorted[i]]
                y_values = [func_val2sorted[i], func_val2sorted[i + 1]]
                plt.plot(x_values, y_values, color='blue')
                x_values = [func_val1sorted[i], func_val1sorted[i + 1]]
                y_values = [func_val2sorted[i + 1], func_val2sorted[i + 1]]
                plt.plot(x_values, y_values, color='blue') """
    if xlabel != None and ylabel!=None:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    ax.legend()
    plt.grid(True, linewidth=0.5, color='gray', linestyle='-')

    if title is not None:
        plt.title(title)
    if save_filename is not None:
        plt.savefig(os.path.join(os.path.join(os.getcwd(),"results","graphs"),save_filename+".png"),dpi=800,bbox_inches='tight')#plt.savefig(r"C:\Users\korkm\OneDrive\Masaüstü\vectopt_fixed\graphs\sentetikdeneme")
    #plt.show()



def plot_regions(design_points):
    fig, ax = plt.subplots()
    n = len(design_points)
    color = iter(cm.rainbow(np.linspace(0, 1, n)))

    lower = -12
    upper = 12
    for point in design_points:
        c = next(color)
        p = Polygon(point.R.get_vertices(), facecolor=c, alpha=0.7, label=str(point.x))

        ax.add_patch(p)

    ax.set_xlim([lower, upper])
    ax.set_ylim([lower, upper])

    plt.grid()
    plt.title("plot")
    plt.legend()
    plt.show()
