def check_params(features_spec, groups_spec, num_bins, edge_range_spec, trim_outliers, trim_percentile):
    """Necessary check on values, ranges, and types."""

    if isinstance(features_spec, str) and isinstance(groups_spec, str):
        features, groups = read_features_and_groups(features_spec, groups_spec)
    else:
        features, groups = features_spec, groups_spec

    num_bins, edge_range, features, groups = type_cast_params(num_bins, edge_range_spec, features, groups)
    num_values = len(features)

    # memberships
    group_ids, num_groups = identify_groups(groups)
    num_links = np.int64(num_groups * (num_groups - 1) / 2.0)

    check_param_ranges(num_bins, num_groups, num_values, trim_outliers, trim_percentile)

    return features, groups, num_bins, edge_range, group_ids, num_groups, num_links