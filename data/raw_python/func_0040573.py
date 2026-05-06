def weighted_rand(weights, round_result=False):
    """
    Generate a non-uniform random value based on a list of weight tuples.

    Treats weights as coordinates for a probability distribution curve and
    rolls accordingly. Constructs a piece-wise linear curve according to
    coordinates given in ``weights`` and rolls random values in the
    curve's bounding box until a value is found under the curve

    Weight tuples should be of the form: (outcome, strength).

    Args:
        weights: (list): the list of weights where each weight
            is a tuple of form ``(float, float)`` corresponding to
            ``(outcome, strength)``.
            Weights with strength ``0`` or less will have no chance to be
            rolled. The list must be sorted in increasing order of outcomes.
        round_result (bool): Whether or not to round the resulting value
            to the nearest integer.

    Returns:
        float: A weighted random number

        int: A weighted random number rounded to the nearest ``int``

    Example:
        >>> weighted_rand([(-3, 4), (0, 10), (5, 1)])          # doctest: +SKIP
        -0.650612268193731
        >>> weighted_rand([(-3, 4), (0, 10), (5, 1)])          # doctest: +SKIP
        -2
    """
    # If just one weight is passed, simply return the weight's name
    if len(weights) == 1:
        return weights[0][0]

    # Is there a way to do this more efficiently? Maybe even require that
    # ``weights`` already be sorted?
    weights = sorted(weights, key=lambda w: w[0])

    x_min = weights[0][0]
    x_max = weights[-1][0]
    y_min = 0
    y_max = max([point[1] for point in weights])

    # Roll random numbers until a valid one is found
    attempt_count = 0
    while attempt_count < 500000:
        # Get sample point
        sample = (random.uniform(x_min, x_max), random.uniform(y_min, y_max))
        if _point_under_curve(weights, sample):
            # The sample point is under the curve
            if round_result:
                return int(round(sample[0]))
            else:
                return sample[0]
        attempt_count += 1
    else:
        warnings.warn(
             'Point not being found in weighted_rand() after 500000 '
             'attempts, defaulting to a random weight point. '
             'If this happens often, it is probably a bug.')
        return random.choice(weights)[0]