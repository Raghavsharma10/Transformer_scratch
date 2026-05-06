def _kmeans_init(X, n_clusters, method='balanced', rng=None):
    """ Initialize k=n_clusters centroids randomly
    """
    n_samples = X.shape[0]
    if rng is None:
        cent_idx = np.random.choice(n_samples, replace=False, size=n_clusters)
    else:
        #print('Generate random centers using RNG')
        cent_idx = rng.choice(n_samples, replace=False, size=n_clusters)
    
    centers = X[cent_idx,:]
    mean_X = np.mean(X, axis=0)
    
    if method == 'balanced':
        centers[n_clusters-1] = n_clusters*mean_X - np.sum(centers[:(n_clusters-1)], axis=0)
    
    return (centers)