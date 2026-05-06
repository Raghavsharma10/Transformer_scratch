def duration_distance(item_a, item_b, max_value):
    """
    Absolute difference in the duration of two items

    Args:
        item_a: STObject from the first set in TrackMatcher
        item_b: STObject from the second set in TrackMatcher
        max_value: Maximum distance value used as scaling value and upper constraint.

    Returns:
        Distance value between 0 and 1.
    """
    duration_a = item_a.times.size
    duration_b = item_b.times.size
    return np.minimum(np.abs(duration_a - duration_b), max_value) / float(max_value)