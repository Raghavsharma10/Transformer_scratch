def bound_weights(weights, minimum=None, maximum=None):
    """
    Bound a weight list so that all outcomes fit within specified bounds.

    The probability distribution within the ``minimum`` and ``maximum``
    values remains the same. Weights in the list with outcomes outside of
    ``minimum`` and ``maximum`` are removed.
    If weights are removed from either end, attach weights at the modified
    edges at the same weight (y-axis) position they had interpolated in the
    original list.

    If neither ``minimum`` nor ``maximum`` are set, ``weights`` will be
    returned unmodified. If both are set, ``minimum`` must be less
    than ``maximum``.

    Args:
        weights (list): the list of weights where each weight
            is a ``tuple`` of form ``(float, float)`` corresponding to
            ``(outcome, weight)``. Must be sorted in increasing order
            of outcomes
        minimum (float): Lowest allowed outcome for the weight list
        maximum (float): Highest allowed outcome for the weight list

    Returns:
        list: A list of 2-tuples of form ``(float, float)``,
        the bounded weight list.

    Raises:
        ValueError: if ``maximum < minimum``

    Example:
        >>> weights = [(0, 0), (2, 2), (4, 0)]
        >>> bound_weights(weights, 1, 3)
        [(1, 1), (2, 2), (3, 1)]
    """
    # Copy weights to avoid side-effects
    bounded_weights = weights[:]
    # Remove weights outside of minimum and maximum
    if minimum is not None and maximum is not None:
        if maximum < minimum:
            raise ValueError
        bounded_weights = [bw for bw in bounded_weights
                           if minimum <= bw[0] <= maximum]
    elif minimum is not None:
        bounded_weights = [bw for bw in bounded_weights
                           if minimum <= bw[0]]
    elif maximum is not None:
        bounded_weights = [bw for bw in bounded_weights
                           if bw[0] <= maximum]
    else:
        # Both minimum and maximum are None - the bound list is the same
        # as the original
        return bounded_weights
    # If weights were removed, attach new endpoints where they would have
    # appeared in the original curve
    if (bounded_weights[0][0] > weights[0][0] and
            bounded_weights[0][0] != minimum):
        bounded_weights.insert(0, (minimum, _linear_interp(weights, minimum)))
    if (bounded_weights[-1][0] < weights[-1][0] and
            bounded_weights[-1][0] != maximum):
        bounded_weights.append((maximum, _linear_interp(weights, maximum)))
    return bounded_weights