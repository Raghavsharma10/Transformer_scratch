def _kmeans_run(X, n_clusters, max_iter, tol):
    """ Run a single trial of k-means clustering
        on dataset X, and given number of clusters
    """
    membs = np.empty(shape=X.shape[0], dtype=int)
    centers = _kmeans_init(X, n_clusters)

    sse_last = 9999.9
    n_iter = 0
    for it in range(1,max_iter):
        membs = _assign_clusters(X, centers)
        centers,sse_arr = _update_centers(X, membs, n_clusters)
        sse_total = np.sum(sse_arr)
        if np.abs(sse_total - sse_last) < tol:
            n_iter = it
            break
        sse_last = sse_total

    return(centers, membs, sse_total, sse_arr, n_iter)