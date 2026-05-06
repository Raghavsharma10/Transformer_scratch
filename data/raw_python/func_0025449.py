def centroid_distance(item_a, time_a, item_b, time_b, max_value):
    """
    Euclidean distance between the centroids of item_a and item_b.

    Args:
        item_a: STObject from the first set in ObjectMatcher
        time_a: Time integer being evaluated
        item_b: STObject from the second set in ObjectMatcher
        time_b: Time integer being evaluated
        max_value: Maximum distance value used as scaling value and upper constraint.

    Returns:
        Distance value between 0 and 1.
    """
    ax, ay = item_a.center_of_mass(time_a)
    bx, by = item_b.center_of_mass(time_b)
    return np.minimum(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2), max_value) / float(max_value)