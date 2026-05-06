def area_difference(item_a, time_a, item_b, time_b, max_value):
    """
    RMS Difference in object areas.

    Args:
        item_a: STObject from the first set in ObjectMatcher
        time_a: Time integer being evaluated
        item_b: STObject from the second set in ObjectMatcher
        time_b: Time integer being evaluated
        max_value: Maximum distance value used as scaling value and upper constraint.

    Returns:
        Distance value between 0 and 1.
    """
    size_a = item_a.size(time_a)
    size_b = item_b.size(time_b)
    diff = np.sqrt((size_a - size_b) ** 2)
    return np.minimum(diff, max_value) / float(max_value)