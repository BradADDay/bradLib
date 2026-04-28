
import matplotlib.pyplot as plt
import sympy as sy

# =============================================================================
# Matplotlib utility
# =============================================================================

class plotter():
    """
    A generaliseable plotting class
    """
    
    # Creation of the figure and axis objects ---------------------------------
    
    def __init__(self, title = None, figsize=(8,5)):
        # Creating the figure and axes objects, setting title
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_title(title)
    
    # General utility methods -------------------------------------------------
    
    def save(self, name):
        self.fig.savefig(name)
    
    # Plotting methods --------------------------------------------------------
    
    def plot(self, data, xyLabels = [None, None], label=None, legendLoc="best"):
        xData, yData = data
        self.ax.plot(xData, yData, label=label) # Plotting the data as a line
        self.ax.set_xlabel(xyLabels[0])         # Setting the x and y axis labels
        self.ax.set_ylabel(xyLabels[1])
        if label != None:
            self.ax.legend(loc=legendLoc)       # Applying a legend to the plot
        
    def scatter(self, data, xyLabels = [None, None], label=None, legendLoc="best", colour=None, markerSize=10, markerStyle = "o"):
        xData, yData = data
        self.ax.scatter(xData, yData, label=label, c=colour, s=markerSize, marker=markerStyle)  # Plotting the data as a scatter
        self.ax.set_xlabel(xyLabels[0])             # Setting the x, y axis labels
        self.ax.set_ylabel(xyLabels[1])
        if label != None:
            self.ax.legend(loc=legendLoc)           # Applying a legend to the plot
    
    def errorbar(self, data, errorbars):
        # Plot errobars
        self.ax.errorbar(data[0], data[1], xerr=errorbars[0], yerr=errorbars[1], fmt="none", capsize=5, elinewidth=2, markeredgewidth=2)
    
    # Formatting methods ------------------------------------------------------
    
    def scales(self, xScale, yScale):
        self.ax.set_xscale(xScale)
        self.ax.set_yscale(yScale)
        
    def limits(self, xLimitD=None, xLimitU=None, yLimitD=None, yLimitU=None):
        # Setting the x and y axes limits
        self.ax.set_xlim(xLimitD, xLimitU)  
        self.ax.set_ylim(yLimitD, yLimitU)
        
    def grid(self, c="k"):
        # Setting the grid of the plot
        self.ax.grid(which='major', color=c, linestyle='-', linewidth=0.6, alpha = 0.6)
        self.ax.grid(which='minor', color=c, linestyle='--', linewidth=0.4, alpha = 0.4)
        self.ax.minorticks_on()     # Turning on minorticks
        self.ax.set_axisbelow(True) # Setting the axis below the data
        
    def aspect(self, aspectRatio="equal"):
        self.ax.set_aspect(aspectRatio) # Setting the aspect ratio of the axes
        
    def invert(self, axis="both"):
        # Invert the axes
        if axis == "both":
            plt.gca().invert_xaxis()
            plt.gca().invert_yaxis()
        if axis == "x":
            plt.gca().invert_xaxis()
        if axis == "y":
            plt.gca().invert_yaxis()
            
    def imshow(self, data, cmap="gray"):
        # Plot an imshow
        im = self.ax.imshow(data, cmap=cmap)
        self.fig.colorbar(im, ax=self.ax)
        
    def legend(self):
        # Applies a legend to the axis
        self.ax.legend()
    
    # Defaults ----------------------------------------------------------------
    
    def defaultPlot(self, data, xyLabels, label=None):
        # Default parameters for quick line plotting
        self.plot(data, xyLabels, label)
        self.grid()
    
    def defaultScatter(self, data, xyLabels, label=None):
        # Default parameters for quick scatter plotting
        self.scatter(data, xyLabels, label)
        self.grid()
        
    # -------------------------------------------------------------------------
