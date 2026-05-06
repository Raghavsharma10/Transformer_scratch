def type_cast_params(num_bins, edge_range_spec, features, groups):
    """Casting inputs to required types."""

    if isinstance(num_bins, str):
        # possible when called from CLI
        num_bins = np.float(num_bins)

    # rounding it to ensure it is int
    num_bins = np.rint(num_bins)

    if np.isnan(num_bins) or np.isinf(num_bins):
        raise ValueError('Invalid value for number of bins! Choose a natural number >= {}'.format(minimum_num_bins))

    if edge_range_spec is None:
        edge_range = edge_range_spec
    elif isinstance(edge_range_spec, collections.Sequence):
        if len(edge_range_spec) != 2:
            raise ValueError('edge_range must be a tuple of two values: (min, max)')
        if edge_range_spec[0] >= edge_range_spec[1]:
            raise ValueError('edge_range : min {} is not less than the max {} !'.format(edge_range_spec[0], edge_range_spec[1]))
        if not np.all(np.isfinite(edge_range_spec)):
            raise ValueError('Infinite or NaN values in edge range : {}'.format(edge_range_spec))

        # converting it to tuple to make it immutable
        edge_range = tuple(edge_range_spec)
    else:
        raise ValueError('Invalid edge range! Must be a tuple of two values (min, max)')

    if not isinstance(features, np.ndarray):
        features = np.array(features)

    if not isinstance(groups, np.ndarray):
        groups = np.array(groups)

    return num_bins, edge_range, features, groups