def start_time_distance(item_a, item_b, max_value):
    """
    Absolute difference between the starting times of each item.

    Args:
        item_a: STObject from the first set in TrackMatcher
        item_b: STObject from the second set in TrackMatcher
        max_value: Maximum distance value used as scaling value and upper constraint.

    Returns:
        Distance value between 0 and 1.
    """
    start_time_diff = np.abs(item_a.times[0] - item_b.times[0])
    return np.minimum(start_time_diff, max_value) / float(max_value)