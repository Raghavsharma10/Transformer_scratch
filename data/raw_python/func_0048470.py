def preprocess_histogram(hist, values, edges):
    """Handles edge-cases and extremely-skewed histograms"""

    # working with extremely skewed histograms
    if np.count_nonzero(hist) == 0:
        # all of them above upper bound
        if np.all(values >= edges[-1]):
            hist[-1] = 1
        # all of them below lower bound
        elif np.all(values <= edges[0]):
            hist[0] = 1

    return hist