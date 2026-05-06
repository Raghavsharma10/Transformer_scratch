def make_random_histogram(length=100, num_bins=10):
    "Returns a sequence of histogram density values that sum to 1.0"

    hist, bin_edges = np.histogram(np.random.random(length),
                                   bins=num_bins, density=True)

    # to ensure they sum to 1.0
    hist = hist / sum(hist)

    if len(hist) < 2:
        raise ValueError('Invalid histogram')

    return hist