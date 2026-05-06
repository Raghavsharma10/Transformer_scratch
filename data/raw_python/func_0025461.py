def mean_area_distance(item_a, item_b, max_value):
    """
    Absolute difference in the means of the areas of each track over time.

    Args:
        item_a: STObject from the first set in TrackMatcher
        item_b: STObject from the second set in TrackMatcher
        max_value: Maximum distance value used as scaling value and upper constraint.

    Returns:
        Distance value between 0 and 1.
    """
    mean_area_a = np.mean([item_a.size(t) for t in item_a.times])
    mean_area_b = np.mean([item_b.size(t) for t in item_b.times])
    return np.abs(mean_area_a - mean_area_b) / float(max_value)