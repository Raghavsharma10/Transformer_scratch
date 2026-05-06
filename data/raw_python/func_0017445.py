def translation(first, second):
    """Create an index of mapped letters (zip to dict)."""
    if len(first) != len(second):
        raise WrongLengthException('The lists are not of the same length!')
    return dict(zip(first, second))