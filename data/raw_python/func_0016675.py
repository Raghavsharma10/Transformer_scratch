def strip_head(sequence, values):
    """Strips elements of `values` from the beginning of `sequence`."""
    values = set(values)
    return list(itertools.dropwhile(lambda x: x in values, sequence))