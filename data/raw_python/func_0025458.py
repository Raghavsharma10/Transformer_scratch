def start_centroid_distance(item_a, item_b, max_value):
    """
    Distance between the centroids of the first step in each object.

    Args:
        item_a: STObject from the first set in TrackMatcher
        item_b: STObject from the second set in TrackMatcher
        max_value: Maximum distance value used as scaling value and upper constraint.

    Returns:
        Distance value between 0 and 1.
    """
    start_a = item_a.center_of_mass(item_a.times[0])
    start_b = item_b.center_of_mass(item_b.times[0])
    start_distance = np.sqrt((start_a[0] - start_b[0]) ** 2 + (start_a[1] - start_b[1]) ** 2)
    return np.minimum(start_distance, max_value) / float(max_value)