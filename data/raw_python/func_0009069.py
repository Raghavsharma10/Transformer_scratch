def make_interactive_tree(matrix=None,labels=None):
    '''make interactive tree will return complete html for an interactive tree
    :param title: a title for the plot, if not defined, will be left out.
    '''
    from scipy.cluster.hierarchy import (
        dendrogram, 
        linkage,
        to_tree
    )

    d3 = None
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist

    if isinstance(matrix,pandas.DataFrame):
        Z = linkage(matrix, 'ward') # clusters
        T = to_tree(Z, rd=False)

        if labels == None:
            labels = matrix.index.tolist()
        lookup = dict(zip(range(len(labels)), labels))

        # Create a dendrogram object without plotting
        dend = dendrogram(Z,no_plot=True,
                      orientation="right",
                      leaf_rotation=90.,  # rotates the x axis labels
                      leaf_font_size=8.,  # font size for the x axis labels
                      labels=labels)

        d3 = dict(children=[], name="root")
        add_node(T, d3)
        label_tree(d3["children"][0],lookup)
    else:
        bot.warning('Please provide data as pandas Data Frame.')
    return d3