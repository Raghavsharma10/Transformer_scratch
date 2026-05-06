def mean_minimum_centroid_distance(item_a, item_b, max_value):
    """
    RMS difference in the minimum distances from the centroids of one track to the centroids of another track

    Args:
        item_a: STObject from the first set in TrackMatcher
        item_b: STObject from the second set in TrackMatcher
        max_value: Maximum distance value used as scaling value and upper constraint.

    Returns:
        Distance value between 0 and 1.
    """
    centroids_a = np.array([item_a.center_of_mass(t) for t in item_a.times])
    centroids_b = np.array([item_b.center_of_mass(t) for t in item_b.times])
    distance_matrix = (centroids_a[:, 0:1] - centroids_b.T[0:1]) ** 2 + (centroids_a[:, 1:] - centroids_b.T[1:]) ** 2
    mean_min_distances = np.sqrt(distance_matrix.min(axis=0).mean() + distance_matrix.min(axis=1).mean())
    return np.minimum(mean_min_distances, max_value) / float(max_value)