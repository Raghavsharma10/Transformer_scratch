def _kmedoids_run(X, n_clusters, distance, max_iter, tol, rng):
    """ Run a single trial of k-medoids clustering
        on dataset X, and given number of clusters
    """
    membs = np.empty(shape=X.shape[0], dtype=int)
    centers = kmeans._kmeans_init(X, n_clusters, method='', rng=rng)

    sse_last = 9999.9
    n_iter = 0
    for it in range(1,max_iter):
        membs = kmeans._assign_clusters(X, centers)
        centers,sse_arr = _update_centers(X, membs, n_clusters, distance)
        sse_total = np.sum(sse_arr)
        if np.abs(sse_total - sse_last) < tol:
            n_iter = it
            break
        sse_last = sse_total

    return(centers, membs, sse_total, sse_arr, n_iter)