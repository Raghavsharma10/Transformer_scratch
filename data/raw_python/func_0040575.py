def weighted_order(weights):
    """
    Non-uniformally order a list according to weighted priorities.

    ``weights`` is a list of tuples of form ``(Any, float or int)``
    corresponding to ``(item, strength)``. The output list is constructed
    by repeatedly calling ``weighted_choice()`` on the weights, adding items
    to the end of the list as they are picked.

    Higher strength weights will have a higher chance of appearing near the
    beginning of the output list.

    A list weights with uniform strengths is equivalent to calling
    ``random.shuffle()`` on the list of items.

    If any weight strengths are ``<= 0``, a ``ProbabilityUndefinedError``
    is be raised.

    Passing an empty list will return an empty list.

    Args:
        weights (list): a list of tuples of form ``(Any, float or int)``
            corresponding to ``(item, strength)``. The output list is
            constructed by repeatedly calling ``weighted_choice()`` on
            the weights, appending items to the output list
            as they are picked.

    Returns:
        list: the newly ordered list

    Raises:
        ProbabilityUndefinedError: if any weight's strength is below 0.

    Example:
        >>> weights = [('Probably Earlier', 100),
        ...            ('Probably Middle', 20),
        ...            ('Probably Last', 1)]
        >>> weighted_order(weights)                            # doctest: +SKIP
        ['Probably Earlier', 'Probably Middle', 'Probably Last']
    """
    if not len(weights):
        return []
    if any(w[1] <= 0 for w in weights):
        raise ProbabilityUndefinedError(
            'All weight values must be greater than 0.')
    working_list = weights[:]
    output_list = []
    while working_list:
        picked_item = weighted_choice(working_list,
                                      as_index_and_value_tuple=True)
        output_list.append(picked_item[1])
        del working_list[picked_item[0]]
    return output_list