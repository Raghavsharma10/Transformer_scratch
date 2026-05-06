def max_intensity(item_a, time_a, item_b, time_b, max_value):
    """
    RMS difference in maximum intensity

    Args:
        item_a: STObject from the first set in ObjectMatcher
        time_a: Time integer being evaluated
        item_b: STObject from the second set in ObjectMatcher
        time_b: Time integer being evaluated
        max_value: Maximum distance value used as scaling value and upper constraint.

    Returns:
        Distance value between 0 and 1.
    """
    intensity_a = item_a.max_intensity(time_a)
    intensity_b = item_b.max_intensity(time_b)
    diff = np.sqrt((intensity_a - intensity_b) ** 2)
    return np.minimum(diff, max_value) / float(max_value)