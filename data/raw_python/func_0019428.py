def _update_centers(X, membs, n_clusters):
    """ Update Cluster Centers:
           calculate the mean of feature vectors for each cluster
    """
    centers = np.empty(shape=(n_clusters, X.shape[1]), dtype=float)
    sse = np.empty(shape=n_clusters, dtype=float)
    for clust_id in range(n_clusters):
        memb_ids = np.where(membs == clust_id)[0]

        if memb_ids.shape[0] == 0:
            memb_ids = np.random.choice(X.shape[0], size=1)
            #print("Empty cluster replaced with ", memb_ids)
        centers[clust_id,:] = np.mean(X[memb_ids,:], axis=0)
        
        sse[clust_id] = _cal_dist2center(X[memb_ids,:], centers[clust_id,:]) 
    return(centers, sse)