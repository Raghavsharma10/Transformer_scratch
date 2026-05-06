def all_pairs_normalized_distances_reference(X):
    """
    Reference implementation of normalized all-pairs distance, used
    for testing the more efficient implementation above for equivalence.
    """
    n_samples, n_cols = X.shape
    # matrix of mean squared difference between between samples
    D = np.ones((n_samples, n_samples), dtype="float32") * np.inf
    for i in range(n_samples):
        diffs = X - X[i, :].reshape((1, n_cols))
        missing_diffs = np.isnan(diffs)
        missing_counts_per_row = missing_diffs.sum(axis=1)
        valid_rows = missing_counts_per_row < n_cols
        D[i, valid_rows] = np.nanmean(
            diffs[valid_rows, :] ** 2,
            axis=1)
    return D