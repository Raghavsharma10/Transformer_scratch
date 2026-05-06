def _add_tree_node(tree, label, ilev,  X=None, size=None, center=None, sse=None, parent=None):
    """ Add a node to the tree
         if parent is not known, the node is a root

        The nodes of this tree keep properties of each cluster/subcluster:
           size   --> cluster size as the number of points in the cluster
           center --> mean of the cluster
           label  --> cluster label
           sse    --> sum-squared-error for that single cluster
           ilev   --> the level at which this node is split into 2 children
    """
    if size is None:
        size = X.shape[0]
    if (center is None):
        center = np.mean(X, axis=0)
    if (sse is None):
        sse = _kmeans._cal_dist2center(X, center)

    center = list(center)
    datadict = {
        'size'  : size,
        'center': center, 
        'label' : label, 
        'sse'   : sse,
        'ilev'  : None 
    }
    if (parent is None):
        tree.create_node(label, label, data=datadict)
    else:
        tree.create_node(label, label, parent=parent, data=datadict)
        tree.get_node(parent).data['ilev'] = ilev

    return(tree)