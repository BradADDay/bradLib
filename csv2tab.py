# =============================================================================
# 
# Importing Libraries and Defining Plotting Parameters
# 
# =============================================================================

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sklearn as sk
import time
 
# Defining rcParams for plotting
plt.rcParams['axes.formatter.limits'] = [-2, 2]
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'

def numberInput(message, inputType=float):
    """
    Function to prevent user input from causing a type error
    
    Parameters
    ----------
    message : The message to prompt user input.
    inputType : The variable type you wish the user to enter, optional. The default is float.
    
    Returns
    ----------
    Returns: The users input in the specified type
    """
    while True:
        try:
            return inputType (input(message))
        except: 
            print('\nInvalid entry, please try again')
            pass

def plotting(x, y, xyLabels, types, labels=None, xExtents = None, yExtents = None, grid = True, 
             title = None, sharex = False, thickness = 1, yScale="linear", 
             xScale = "linear", invertY = False, yTicks = None):
    
    """
    A generaliseable 2D plotting function
    
    Parameters
    ----------
    x: array-like
        A 2-dimensional array containing the data to be plotted on the x axis
    y: array-like
        A 2-dimensional array containing the data to be plotted on the y-axis
    xyLabels: list
        The x and y axis labels in a list of length 2
    types: list or string
        The type of plot to correlating each set of data, can be "p" or "s" for plot or scatter
        If only one is entered all plots will follow the entered type
    labels: optional, list
         The labels for each dataset, the default is None
    xExtents: optional, array-like
        The upper and lower bound for the x-axis, the default is None
    yExtents: optional, array-like
        The upper and lower bound for the y-axis, the default is None
    grid: optional, boolean
        Whether to plot a grid, the default is True
    title: optional, string
        The title for the plot, the default is None
    sharex: optional, boolean
        Whether to use just one dataset for x, the default is False
    thickness: optional, float
        The linethickness or point size for the plot, the default is 1
    yScale: optional, string
        The type of scale to use on the y-axis, the default is "linear"
    xScale: optional, string
        The type of scale to use on the x-axis, the default is "linear"
    invertY: optional, boolean
        Whether to plot the y axis as descending, the default is False
    yTicks: optional, array-like
        The option to manually set the ticks on the y-axis
    
    Returns
    ----------
    Returns: The 2D plot of the data
    """
    
    # Creating the subplot
    fig, ax = plt.subplots()
    
    # Checking if the labels were entered as a list, creating a list if not and if no labels entered
    if type(labels) == str or labels == None:
        labelVal = labels
        labels = []
        for i in range(len(y)):
            labels.append(labelVal)
    
    # Checking if the types were entered as a list, creating a list if not
    if type(types) == str:
        typeVal = types
        types = []
        for i in range(len(y)):
            types.append(typeVal)
    
    # Allowing the input of one x array if desired
    if sharex == True:
        xs=[x]
        for i in range(len(y)):
            xs.append(x)
        x = xs
            
    # Looping through the data to be plot and checking the plottype
    for i in range(len(y)):
        
        # Plotting as a line plot
        if types[i] == 'p':
            ax.plot(x[i], y[i], label = labels[i], lw=thickness)
        
        # Plotting as a scatter plot
        elif types[i] == 's':
            ax.scatter(x[i], y[i], s=thickness, label = labels[i])
        
    # Setting the axes below the data
    ax.set_axisbelow(True)
    
    # Labelling the axes
    ax.set_xlabel(xyLabels[0])
    ax.set_ylabel(xyLabels[1])
    
    # Setting the title
    ax.set_title(title)
    
    # Setting the axis scale types
    ax.set_yscale(yScale)
    ax.set_xscale(xScale)
    
    # 
    if yTicks != None:
        ax.set_yticks(yTicks[0], yTicks[1])
    
    # Checking whether to plot a grid
    if grid == True:
        ax.grid(which='major', color='k', linestyle='-', linewidth=0.6, alpha = 0.6)
        ax.grid(which='minor', color='k', linestyle='--', linewidth=0.4, alpha = 0.4)
        ax.minorticks_on()
    
    # Setting the plotting extents if they are specified
    if xExtents != None:
        ax.set_xlim(xExtents[0], xExtents[1])
    if yExtents != None:
        ax.set_ylim(yExtents[0], yExtents[1])
        
    # Applying legend if true
    if labels[0] != None:
        ax.legend(markerscale = 10)
    
    # Inverting the direction of the y-axis if True
    if invertY == True:
        plt.gca().invert_yaxis() 
    
    plt.show()

def dataFiltering(dataFrame):
    """
    Filtering the dataFrame for NaN, negative parallaxes, large parallax errors and extreme values
    
    Parameters
    ----------
    path : string
        The filepath for the dataFrame to be filtered.

    Returns
    -------
    filteredDataframe : pandas.core.frame.DataFrame
        The filtered dataFrame in a dataFrame format
    varList : list
        A list of all varList stored in the dataFrame
    """
    
    # Removing all rows with NaN
    filteredDataframe = dataFrame.dropna()
    
    # Reading the column headers to produce a list of the varList and another without the materials (non-numeric dataFrame)
    varList = list(filteredDataframe.columns)
    
    # Ensuring all parallax dataFrame is positive
    filteredDataframe = filteredDataframe[(filteredDataframe["parallax"] >= 0)]
    
    # Ensuring all parallax errors are less than 20% relative to the parallax
    parallaxFilter = np.where(filteredDataframe.parallax_error <= 0.2 * filteredDataframe.parallax)
    filteredDataframe = filteredDataframe.iloc[parallaxFilter]
    
    # Ensuring all proper motion Ascensions and Declinations are within 6 standard deviations
    radius = 50
    extremityFilter = np.where((filteredDataframe["pmra"]-np.mean(filteredDataframe.pmra))**2 + (filteredDataframe["pmdec"]-np.mean(filteredDataframe.pmdec))**2 <= radius**2)
    filteredDataframe = filteredDataframe.iloc[extremityFilter]
    
    # Removing duplicate rows
    filteredDataframe.drop_duplicates(inplace=True)
    
    # Resetting the index of the dataFrame such that it fits the filtered dataFrame
    filteredDataframe = filteredDataframe.reset_index(drop=True)
    
    return(filteredDataframe, varList)

def scaling(dataFrame, scaler):
    """
    Takes in a dataFrame and returns a version scaled based on the features
    Parameters
    ----------
    dataFrame : pandas.DataFrame
        The dataFrame to be scaled.
    scaler : function
        The scaling function to be used.

    Returns
    -------
    scaledDataframe : pandas.DataFrame
        The scaled dataFrame.
    """
    scaler.fit(dataFrame)
    
    scaledDataframe = scaler.transform(dataFrame)
    
    scaledDataframe = pd.DataFrame(scaledDataframe, columns = dataFrame.columns)
    
    return scaledDataframe

def kMeans(dataFrame, clusters):
    """
    Clustering the dataFrame using the K-Means method
    Parameters
    ----------
    dataFrame : pandas.dataframe
        The 2D dataset to be clustered.
    clusters : int
        The number of clusters to produce.

    Returns
    -------
    clusteredX : array-like
        a 2D array containing the x data separated into clusters.
    clusteredY : array-like
        a 2D array containing the y data separated into clusters.
    uniqueIndices : array-like
        The cluster indices, used for selection and labelling within plotting.
    clusterIndices : array-like
        The list of cluster indices correlating to each entry in the dataFrame.

    """
    # Converting to an array
    dataFrame = np.array(dataFrame)
    # Clustering
    kmeans = sk.cluster.KMeans(clusters, random_state=0).fit(dataFrame)
    # Taking the Indices and creating a list of the unique values
    clusterIndices = kmeans.labels_
    uniqueIndices = np.unique(clusterIndices)
    
    # Creating lists for storage
    clusteredX = []
    clusteredY = []
    
    # Grouping the data in 2D arrays based on its cluster
    for cluster in uniqueIndices:
        clusteredX.append(dataFrame[np.where(clusterIndices == cluster),0])
        clusteredY.append(dataFrame[np.where(clusterIndices == cluster),1])
    
    return clusteredX, clusteredY, uniqueIndices, clusterIndices
        
def gaussMix(dataFrame, clusters):
    """
    Clustering the dataFrame using the Gaussian Mixture method
    Parameters
    ----------
    dataFrame : pandas.dataframe
        The 2D dataset to be clustered.
    clusters : int
        The number of clusters to produce.

    Returns
    -------
    clusteredX : array-like
        a 2D array containing the x data separated into clusters.
    clusteredY : array-like
        a 2D array containing the y data separated into clusters.
    uniqueIndices : array-like
        The cluster indices, used for selection and labelling within plotting.
    clusterIndices : array-like
        The list of cluster indices correlating to each entry in the dataFrame.

    """
    # Converting to an array
    dataFrame = np.array(dataFrame)
    print(dataFrame)
    # Clustering
    gaussMix = sk.mixture.GaussianMixture(clusters, random_state=0).fit(dataFrame)
    # Taking the Indices and creating a list of the unique values
    clusterIndices = gaussMix.predict(dataFrame)
    uniqueIndices = np.unique(clusterIndices)
    
    # Creating lists for storage
    clusteredX = []
    clusteredY = []
    
    # Grouping the data in 2D arrays based on its cluster
    for cluster in uniqueIndices:
        clusteredX.append(dataFrame[np.where(clusterIndices == cluster),0])
        clusteredY.append(dataFrame[np.where(clusterIndices == cluster),1])
    
    return clusteredX, clusteredY, uniqueIndices, clusterIndices

def pca(dataFrame, scaledData):
    """
    A function to perform principal component analysis
    Parameters
    ----------
    dataFrame : pandas.dataframe
        The unscaled dataframe, used for projection.
    scaledData : pandas.dataframe
        The scaled dataframe, used for the PCA.

    Returns
    -------
    projection : pandas.dataframe
        The two principal components projected onto the dataset.
    variances : array-like
        The variance percentage corresponding to each principal component.
    eigVals : array-like
        The eigenvalues corresponding to the variance of each principal component.
    eigVecs : array-like
        The eigenvector matrix.

    """
    # Generating the Covariance matrix
    covMatrix = np.cov(scaledData.T)
    # Calculating the eigenvalues and eigenvectors
    eigVals, eigVecs = np.linalg.eigh(covMatrix)
    # Taking the two eigenvectors correlating to the largest variance (largest eigenvalues)
    V = eigVecs[:,-2:]
    
    # Projecting the two principal components onto the data
    projection = np.matmul(dataFrame, V)
    
    # Calculating the percentage of the variance produced from each principal component
    variances = []
    for val in eigVals:
        variances.append(val/np.sum(eigVals))
        
    return projection, variances, eigVals, eigVecs

def loadings(eigenvectors, eigenvalues, columns):
    """
    A function to calculate the loadings from the PCA
    Parameters
    ----------
    eigenvectors : array-like
        The eigenvector matrix.
    eigenvalues : array-like
        The eigenvalues from the PCA.
    columns : array-like
        The column headers for the dataset.

    Returns
    -------
    loadingsDF : pandas.dataframe
        The loadings for each principal component with the columns corresponding to the features of the dataset.
    """
    # Creating lists for storage
    loadings = []
    indices = []
    
    # Taking the length of the eigenvalue array for row labelling
    length = len(eigenvalues)
    
    # looping through the Principal components calculating the loadings for each feature
    for i in range(length):
        loadings.append(eigenvectors[:,i] * np.sqrt(eigenvalues[i]))
        indices.append(f"PC{length-i}")
    
    # Putting the loadings into a dataframe format
    loadingsDF = pd.DataFrame(loadings, index = indices, columns=columns).iloc[::-1]
    
    return loadingsDF
    
def pause():
    """
    A QOL function to make the menu feel smoother
    """
    time.sleep(0.75)

def initialisation(rawData, section):
    """
    A quality of life function to re-initialise key variables within each section
    Parameters
    ----------
    rawData : pandas.dataframe
        The dataframe of data straight from the file.
    section : integer
        The section within which the function is called.

    Returns
    -------
    returns : array-like
        The variables to be returned, changes dependant on [section].
    """
    if section >= 2:
        # Filtering the data
        data, variables = dataFiltering(rawData)
        returns = [data, variables]
        
    if section >= 3:
        # Scaling the data
        scaledData = scaling(data, StandardScaler())
        returns = scaledData
        
    if section >= 4:
        # Clustering the data
        gaussMixDec, gaussMixRa, gaussMixLabels, clusterIndices = gaussMix(scaledData[["pmdec", "pmra"]], 4)
        returns = [data, clusterIndices]
        
    if section == 5:
        # Taking the Cluster of Interest
        interestData = data.iloc[np.where(clusterIndices == 2)].reset_index(drop=True).copy()
        
        # Scaling the interest data
        scaledInterestData = scaling(interestData, StandardScaler())
        
        # Conducting Principal Component Analysis
        projection, variances, eigVals, eigVecs = pca(interestData, scaledInterestData)
        returns = [variables, interestData, eigVecs, eigVals]
    
    return returns

choice = 0
separator = "---------------------------------------"

# Reading the data and skipping bad lines (any that produce an error), aswell as defining blank cells as NaN
print("Reading the Data From GitHub...")
rawData = pd.read_csv("https://raw.githubusercontent.com/FTurci/demo-files/refs/heads/main/csv/m45-clean.csv", header=0, on_bad_lines='skip', na_values = '', skipinitialspace=True).copy()

pmDecRaAxisLabels = ["Proper Motion Right Ascension\n(mas/yr)", "Proper Motion Declination\n(mas/yr)"]

while choice != "E":
    
    choice = input(f"""\nWhich Section Would You Like to View?
{separator}
1: Reading and Filtering the Data
2: Standardizing the Data
3: Clustering the Data
4: Principal Component Analysis
5: Projection and Factor Analysis
E: Exit the Program
{separator}
Enter Selection: """).upper()

    if choice == "1":
        # =============================================================================
        # 
        # Reading and Filtering the Data
        # 
        # =============================================================================
        
        # Filtering the datafile
        data, variables = dataFiltering(rawData)
        
        print(f"\nThe data has been filtered from {len(rawData)} to {len(data)} datapoints")
        pause()
        
        # plotting the raw and filtered data to compare
        plotting([rawData.pmra], [rawData.pmdec], pmDecRaAxisLabels, "s", title = "Raw Data")
        plotting([data.pmra], [data.pmdec], pmDecRaAxisLabels, "s", title = "Filtered Data")
    
    elif choice == "2":
        # =============================================================================
        # 
        # Standardizing
        # 
        # =============================================================================
        
        # Filtering the data
        # This is done here in case the user does not select the sections in order
        data, variables = initialisation(rawData, 2)
        
        print("\nScaling the Data...")
        
        # scaling the data
        scaledData = scaling(data, StandardScaler())
        pause()
        
        # plotting the scaled data
        plotting([scaledData.pmra], [scaledData.pmdec], pmDecRaAxisLabels, "s", title = "Scaled and Filtered Data")
        
    elif choice == "3":
        # =============================================================================
        # 
        # Clustering
        # 
        # =============================================================================
        
        # Reading, Filtering and Scaling the Data 
        # This is done here in case the user does not select the sections in order
        scaledData = initialisation(rawData, 3)
        
        # Asking the user to enter a number of clusters to be generated
        clusters = numberInput("\nHow Many Clusters Would You Like to Generate?\nEnter an Integer: ", int)
        
        # Calculating and plotting the clusters through kMeans
        kMeanDec, kMeanRa, kMeanLabels, clusterIndices = kMeans(scaledData[["pmdec", "pmra"]], clusters)
        plotting(kMeanRa, kMeanDec, pmDecRaAxisLabels, "s", thickness=.1, title = "Clustered Data Through K-Means")
        
        # Generating and Plotting the clusters through Gaussian Mixture
        gaussMixDec, gaussMixRa, gaussMixLabels, clusterIndices = gaussMix(scaledData[["pmdec", "pmra"]], clusters)
        plotting(gaussMixRa, gaussMixDec, pmDecRaAxisLabels, "s", thickness = .1, title = "Clustered Data Through Gaussian Mixture")
    
    elif choice == "4":
        # =============================================================================
        # 
        # Principal Component Analysis
        # 
        # =============================================================================
        
        # Reading, Filtering, Scaling and Clustering the Data 
        # This is done here in case the user does not select the sections in order
        data, clusterIndices = initialisation(rawData, 4)
        
        # Finding the data within the cluster of interest and assigning it to a new dataframe
        interestData = data.iloc[np.where(clusterIndices == 2)].reset_index(drop=True).copy()
        
        print(f"\nThe data within the small group of stars has {len(interestData)} datapoints")
        pause()
        
        # Plotting the data of interest
        plotting([interestData.pmra], [interestData.pmdec], pmDecRaAxisLabels, "s", title = "The Small Group of Stars That are of Interest")
        
        # Scaling the data of interest
        scaledInterestData = scaling(interestData, StandardScaler())
        
        # Conducting the principal component analysis on the data
        projection, variances, eigVals, eigVecs = pca(interestData, scaledInterestData)
        print(f"\nThe first and second primary components provide a variance of {variances[-1].round(2)} and {variances[-2].round(2)} respectively")
        pause()
        
        # Plotting the two principal components projected onto the data
        plotting([projection[1]], [projection[0]], ["Principal Component1", "Principal Component 2"], "s", title = "The two principal components projected onto the data")
    
    elif choice == "5":
        # =============================================================================
        # 
        # Projection and Factor Analysis
        # 
        # =============================================================================
        
        # Reading, Filtering, Scaling, Clustering and conducting PCA on the Data 
        # This is done here in case the user does not select the sections in order
        variables, interestData, eigVecs, eigVals = initialisation(rawData, 5)

        # Calculating the loadings from all principal components
        loading = loadings(eigVecs, eigVals, variables)
        print(f"\nThe first two loadings are:\n{loading.iloc[0:2]}")
        pause()
        
        # Taking the three largest contributers to the first two principal components
        PC1Dominate = loading.loc["PC1"].nlargest(3)
        PC2Dominate = loading.loc["PC2"].nlargest(3)
        print(f"\nThe features that dominate Principal Component 1 are:\n{PC1Dominate}")
        pause()
        print(f"\nThe features that dominate Principal Component 2 are:\n{PC2Dominate}")
        pause()
        
        # Defining the extents and ticks for the Hertzsprung-Russell diagram
        yExtents = [3, 25]
        yTicks = [[3, 4, 6, 10, 15, 20, 25], ["3", "4", "6", "10", "15", "20", "25"]]
        
        # Plotting the Hertzsprung-Russell Diagram
        plotting([(interestData.bp-interestData.rp)], [interestData.g], ["Colour Index (BP-RP)", "G-Band Magnitude"], "s", yScale = "log", yExtents = yExtents, invertY = True, yTicks = yTicks, title = "The Hertzsprung-Russel Diagram Generated From the Data")
        
    elif choice == "E":
        print("\nTerminating Program...")
    
    else:
        print("\nInvalid Entry, Please Try Again.")
        pause()