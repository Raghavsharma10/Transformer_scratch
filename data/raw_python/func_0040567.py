def _linear_interp(curve, test_x, round_result=False):
    """
    Take a series of points and interpolate between them at ``test_x``.

    Args:
        curve (list[tuple]): A list of ``(x, y)`` points sorted in
            nondecreasing ``x`` value. If multiple points have the same
            ``x`` value, all but the last will be ignored.
        test_x (float): The ``x`` value to find the ``y`` value of

    Returns:
        float: The ``y`` value of the curve at ``test_x``
        if ``round_result is False``

        int: if ``round_result is True`` or the result is a whole number,
        the ``y`` value of the curve at ``test_x`` rounded to the
        nearest whole number.

    Raises:
        ProbabilityUndefinedError: if ``test_x`` is out of the
            domain of ``curve``

    Example:
        >>> curve = [(0, 0), (2, 1)]
        >>> _linear_interp(curve, 0.5)
        0.25
        >>> _linear_interp(curve, 0.5, round_result=True)
        0
    """
    index = 0
    for index in range(len(curve) - 1):
        # Ignore points which share an x value with the following point
        if curve[index][0] == curve[index + 1][0]:
            continue
        if curve[index][0] <= test_x <= curve[index + 1][0]:
            slope = ((curve[index + 1][1] - curve[index][1]) /
                     (curve[index + 1][0] - curve[index][0]))
            y_intercept = curve[index][1] - (slope * curve[index][0])
            result = (slope * test_x) + y_intercept
            if round_result:
                return int(round(result))
            else:
                if result.is_integer():
                    return int(result)
                else:
                    return result
    else:
        raise ProbabilityUndefinedError