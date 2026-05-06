def _cal_dist2center(X, center):
    """ Calculate the SSE to the cluster center
    """
    dmemb2cen = scipy.spatial.distance.cdist(X, center.reshape(1,X.shape[1]), metric='seuclidean')
    return(np.sum(dmemb2cen))