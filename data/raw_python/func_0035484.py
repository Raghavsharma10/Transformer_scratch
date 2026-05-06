def classify_nearest(self, partial_labels):
    '''Simple semi-supervised classification, by assigning unlabeled vertices
    the label of nearest labeled vertex.

    partial_labels: (n,) array of integer labels, -1 for unlabeled.
    '''
    labels = np.array(partial_labels, copy=True)
    unlabeled = labels == -1
    # compute geodesic distances from unlabeled vertices
    D_unlabeled = self.shortest_path(weighted=True)[unlabeled]
    # set distances to other unlabeled vertices to infinity
    D_unlabeled[:,unlabeled] = np.inf
    # find shortest distances to labeled vertices
    idx = D_unlabeled.argmin(axis=1)
    # apply the label of the closest vertex
    labels[unlabeled] = labels[idx]
    return labels