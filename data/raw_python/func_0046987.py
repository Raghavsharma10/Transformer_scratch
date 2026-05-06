def compute_bin_edges(features, num_bins, edge_range, trim_outliers, trim_percentile, use_orig_distr=False):
    "Compute the edges for the histogram bins to keep it the same for all nodes."

    if use_orig_distr:
        print('Using original distribution (without histogram) to compute edge weights!')
        edges=None
        return edges

    if edge_range is None:
        if trim_outliers:
            # percentiles_to_keep = [ trim_percentile, 1.0-trim_percentile] # [0.05, 0.95]
            edges_of_edges = np.array([np.percentile(features, trim_percentile),
                                       np.percentile(features, 100 - trim_percentile)])
        else:
            edges_of_edges = np.array([np.min(features), np.max(features)])
    else:
        edges_of_edges = edge_range

    # Edges computed using data from all nodes, in order to establish correspondence
    edges = np.linspace(edges_of_edges[0], edges_of_edges[1], num=num_bins, endpoint=True)

    return edges