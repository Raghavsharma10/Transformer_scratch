def ceEvalMutual(cluster_runs, cluster_ensemble = None, verbose = False):
    """Compute a weighted average of the mutual information with the known labels, 
        the weights being proportional to the fraction of known labels.

    Parameters
    ----------
    cluster_runs : array of shape (n_partitions, n_samples)
        Each row of this matrix is such that the i-th entry corresponds to the
        cluster ID to which the i-th sample of the data-set has been classified
        by this particular clustering. Samples not selected for clustering
        in a given round are are tagged by an NaN.

    cluster_ensemble : array of shape (n_samples,), optional (default = None)
        The identity of the cluster to which each sample of the whole data-set 
        belong to according to consensus clustering.
 
    verbose : Boolean, optional (default = False)
        Specifies if status messages will be displayed
        on the standard output.

    Returns
    -------
    unnamed variable : float
        The weighted average of the mutual information between
        the consensus clustering and the many runs from the ensemble
        of independent clusterings on subsamples of the data-set.
    """

    if cluster_ensemble is None:
        return 0.0

    if reduce(operator.mul, cluster_runs.shape, 1) == max(cluster_runs.shape):
        cluster_runs = cluster_runs.reshape(1, -1)

    weighted_average_mutual_information = 0

    N_labelled_indices = 0

    for i in range(cluster_runs.shape[0]):
        labelled_indices = np.where(np.isfinite(cluster_runs[i]))[0]
        N = labelled_indices.size

        x = np.reshape(checkcl(cluster_ensemble[labelled_indices], verbose), newshape = N)
        y = np.reshape(checkcl(np.rint(cluster_runs[i, labelled_indices]), verbose), newshape = N)

        q = normalized_mutual_info_score(x, y)

        weighted_average_mutual_information += q * N
        N_labelled_indices += N

    return float(weighted_average_mutual_information) / N_labelled_indices