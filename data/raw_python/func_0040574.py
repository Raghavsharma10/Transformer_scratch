def weighted_choice(weights, as_index_and_value_tuple=False):
    """
    Generate a non-uniform random choice based on a list of option tuples.

    Treats each outcome as a discreet unit with a chance to occur.

    Args:
        weights (list): a list of options where each option
            is a tuple of form ``(Any, float)`` corresponding to
            ``(outcome, strength)``. Outcome values may be of any type.
            Options with strength ``0`` or less will have no chance to be
            chosen.
        as_index_and_value_tuple (bool): Option to return an ``(index, value)``
            tuple instead of just a single ``value``. This is useful when
            multiple outcomes in ``weights`` are the same and you need to know
            exactly which one was picked.

    Returns:
        Any: If ``as_index_and_value_tuple is False``, any one of the items in
        the outcomes of ``weights``

        tuple (int, Any): If ``as_index_and_value_tuple is True``,
        a 2-tuple of form ``(int, Any)`` corresponding to ``(index, value)``.
        the index as well as value of the item that was picked.

    Example:
        >>> choices = [('choice one', 10), ('choice two', 3)]
        >>> weighted_choice(choices)                           # doctest: +SKIP
        # Often will be...
        'choice one'
        >>> weighted_choice(choices,
        ...                 as_index_and_value_tuple=True)     # doctest: +SKIP
        # Often will be...
        (0, 'choice one')
    """
    if not len(weights):
        raise ValueError('List passed to weighted_choice() cannot be empty.')
    # Construct a line segment where each weight outcome is
    # allotted a length equal to the outcome's weight,
    # pick a uniformally random point along the line, and take
    # the outcome that point corresponds to
    prob_sum = sum(w[1] for w in weights)
    if prob_sum <= 0:
        raise ProbabilityUndefinedError(
            'No item weights in weighted_choice() are greater than 0. '
            'Probability distribution is undefined.')
    sample = random.uniform(0, prob_sum)
    current_pos = 0
    i = 0
    while i < len(weights):
        if current_pos <= sample <= (current_pos + weights[i][1]):
            if as_index_and_value_tuple:
                return (i, weights[i][0])
            else:
                return weights[i][0]
        current_pos += weights[i][1]
        i += 1
    else:
        raise AssertionError('Something went wrong in weighted_choice(). '
                             'Please submit a bug report!')