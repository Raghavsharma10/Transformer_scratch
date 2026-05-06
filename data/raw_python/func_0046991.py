def check_param_ranges(num_bins, num_groups, num_values, trim_outliers, trim_percentile):
    """Ensuring the parameters are in valid ranges."""

    if num_bins < minimum_num_bins:
        raise ValueError('Too few bins! The number of bins must be >= 5')

    if num_values < num_groups:
        raise ValueError('Insufficient number of values in features (< number of nodes), or invalid membership!')

    if trim_outliers:
        if trim_percentile < 0 or trim_percentile >= 100:
            raise ValueError('percentile of tail values to trim must be in the semi-open interval [0,1).')
    elif num_values < 2:
        raise ValueError('too few features to compute minimum and maximum')

    return