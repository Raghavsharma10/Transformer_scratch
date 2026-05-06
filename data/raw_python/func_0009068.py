def make_package_tree(matrix=None,labels=None,width=25,height=10,title=None,font_size=None):
    '''make package tree will make a dendrogram comparing a matrix of packages
    :param matrix: a pandas df of packages, with names in index and columns
    :param labels: a list of labels corresponding to row names, will be
    pulled from rows if not defined
    :param title: a title for the plot, if not defined, will be left out.
    :returns a plot that can be saved with savefig
    '''
    from matplotlib import pyplot as plt
    from scipy.cluster.hierarchy import (
        dendrogram, 
        linkage
    )

    if font_size is None:
        font_size = 8.

    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist

    if not isinstance(matrix,pandas.DataFrame):
        bot.info("No pandas DataFrame (matrix) of similarities defined, will use default.")
        matrix = compare_packages()['files.txt']
        title = 'Docker Library Similarity to Base OS'

    Z = linkage(matrix, 'ward')
    c, coph_dists = cophenet(Z, pdist(matrix))

    if labels == None:
        labels = matrix.index.tolist()

    plt.figure(figsize=(width, height))

    if title != None:
        plt.title(title)

    plt.xlabel('image index')
    plt.ylabel('distance')
    dendrogram(Z,
               leaf_rotation=90.,  # rotates the x axis labels
               leaf_font_size=font_size,  # font size for the x axis labels
               labels=labels)
    return plt