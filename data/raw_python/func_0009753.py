def transform_multiple(sequence, transformations, iterations):
    """Chains a transformation a given number of times.

    Args:
        sequence (str): a string or generator onto which transformations are applied
        transformations (dict): a dictionary mapping each char to the string that is
            substituted for it when the rule is applied
        iterations (int): how many times to repeat the transformation

    Yields:
        str: the next character in the output sequence.
    """
    for _ in range(iterations):
        sequence = transform_sequence(sequence, transformations)
    return sequence