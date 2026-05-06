def ellipse_distance(item_a, time_a, item_b, time_b, max_value):
    """
    Calculate differences in the properties of ellipses fitted to each object.

    Args:
        item_a: STObject from the first set in ObjectMatcher
        time_a: Time integer being evaluated
        item_b: STObject from the second set in ObjectMatcher
        time_b: Time integer being evaluated
        max_value: Maximum distance value used as scaling value and upper constraint.

    Returns:
        Distance value between 0 and 1.
    """
    ts = np.array([0, np.pi])
    ell_a = item_a.get_ellipse_model(time_a)
    ell_b = item_b.get_ellipse_model(time_b)
    ends_a = ell_a.predict_xy(ts)
    ends_b = ell_b.predict_xy(ts)
    distances = np.sqrt((ends_a[:, 0:1] - ends_b[:, 0:1].T) ** 2 + (ends_a[:, 1:] - ends_b[:, 1:].T) ** 2)
    return np.minimum(distances[0, 1], max_value) / float(max_value)