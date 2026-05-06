def compute_histogram(values, edges, use_orig_distr=False):
    """Computes histogram (density) for a given vector of values."""

    if use_orig_distr:
        return values

    # ignoring invalid values: Inf and Nan
    values = check_array(values).compressed()

    hist, bin_edges = np.histogram(values, bins=edges, density=True)
    hist = preprocess_histogram(hist, values, edges)

    return hist