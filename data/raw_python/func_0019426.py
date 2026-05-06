def _assign_clusters(X, centers):
    """ Assignment Step:
           assign each point to the closet cluster center
    """
    dist2cents = scipy.spatial.distance.cdist(X, centers, metric='seuclidean')
    membs = np.argmin(dist2cents, axis=1)

    return(membs)