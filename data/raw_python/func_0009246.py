def _energy_distance_from_distance_matrices(
        distance_xx, distance_yy, distance_xy):
    """Compute energy distance with precalculated distance matrices."""
    return (2 * np.mean(distance_xy) - np.mean(distance_xx) -
            np.mean(distance_yy))