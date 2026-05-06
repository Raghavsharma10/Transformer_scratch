def _kmeans(X, n_clusters, max_iter, n_trials, tol):
    """ Run multiple trials of k-means clustering,
        and outputt he best centers, and cluster labels
    """
    n_samples, n_features = X.shape[0], X.shape[1]

    centers_best = np.empty(shape=(n_clusters,n_features), dtype=float)
    labels_best  = np.empty(shape=n_samples, dtype=int)
    for i in range(n_trials):
        centers, labels, sse_tot, sse_arr, n_iter  = _kmeans_run(X, n_clusters, max_iter, tol)
        if i==0:
            sse_tot_best = sse_tot
            sse_arr_best = sse_arr
            n_iter_best = n_iter
            centers_best = centers.copy()
            labels_best  = labels.copy()
        if sse_tot < sse_tot_best:
            sse_tot_best = sse_tot
            sse_arr_best = sse_arr
            n_iter_best = n_iter
            centers_best = centers.copy()
            labels_best  = labels.copy()

    return(centers_best, labels_best, sse_arr_best, n_iter_best)