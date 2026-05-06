def mean_min_time_distance(item_a, item_b, max_value):
    """
    Calculate the mean time difference among the time steps in each object.

    Args:
        item_a: STObject from the first set in TrackMatcher
        item_b: STObject from the second set in TrackMatcher
        max_value: Maximum distance value used as scaling value and upper constraint.

    Returns:
        Distance value between 0 and 1.
    """
    times_a = item_a.times.reshape((item_a.times.size, 1))
    times_b = item_b.times.reshape((1, item_b.times.size))
    distance_matrix = (times_a - times_b) ** 2
    mean_min_distances = np.sqrt(distance_matrix.min(axis=0).mean() + distance_matrix.min(axis=1).mean())
    return np.minimum(mean_min_distances, max_value) / float(max_value)