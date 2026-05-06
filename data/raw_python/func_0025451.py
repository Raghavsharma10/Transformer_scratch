def closest_distance(item_a, time_a, item_b, time_b, max_value):
    """
    Euclidean distance between the pixels in item_a and item_b closest to each other.

    Args:
        item_a: STObject from the first set in ObjectMatcher
        time_a: Time integer being evaluated
        item_b: STObject from the second set in ObjectMatcher
        time_b: Time integer being evaluated
        max_value: Maximum distance value used as scaling value and upper constraint.

    Returns:
        Distance value between 0 and 1.
    """
    return np.minimum(item_a.closest_distance(time_a, item_b, time_b), max_value) / float(max_value)