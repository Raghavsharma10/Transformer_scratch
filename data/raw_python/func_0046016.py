def make_random_histogram(center=0.0, stdev=default_stdev, length=default_feature_dim, num_bins=default_num_bins):
    "Returns a sequence of histogram density values that sum to 1.0"

    hist, bin_edges = np.histogram(get_distr(center, stdev, length),
                                   range=edge_range, bins=num_bins, density=True)
    # to ensure they sum to 1.0
    hist = hist / sum(hist)

    if len(hist) < 2:
        raise ValueError('Invalid histogram')

    return hist, bin_edges