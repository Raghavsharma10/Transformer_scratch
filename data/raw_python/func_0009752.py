def transform_sequence(sequence, transformations):
    """Applies a given set of substitution rules to the given string or generator.
    
    For more background see: https://en.wikipedia.org/wiki/L-system

    Args:
        sequence (str): a string or generator onto which transformations are applied
        transformations (dict): a dictionary mapping each char to the string that is
            substituted for it when the rule is applied

    Yields:
        str: the next character in the output sequence.

    Examples:
        >>> ''.join(transform_sequence('ABC', {}))
        'ABC'
        >>> ''.join(transform_sequence('ABC', {'A': 'AC', 'C': 'D'}))
        'ACBD'
    """
    for c in sequence:
        for k in transformations.get(c, c):
            yield k