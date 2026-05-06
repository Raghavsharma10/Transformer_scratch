def tuplewrap(value):
    """
    INTENDED TO TURN lists INTO tuples FOR USE AS KEYS
    """
    if isinstance(value, (list, set, tuple) + generator_types):
        return tuple(tuplewrap(v) if is_sequence(v) else v for v in value)
    return unwrap(value),