def nonoverlap(item_a, time_a, item_b, time_b, max_value):
    """
    Percentage of pixels in each object that do not overlap with the other object

    Args:
        item_a: STObject from the first set in ObjectMatcher
        time_a: Time integer being evaluated
        item_b: STObject from the second set in ObjectMatcher
        time_b: Time integer being evaluated
        max_value: Maximum distance value used as scaling value and upper constraint.

    Returns:
        Distance value between 0 and 1.
    """
    return np.minimum(1 - item_a.count_overlap(time_a, item_b, time_b), max_value) / float(max_value)