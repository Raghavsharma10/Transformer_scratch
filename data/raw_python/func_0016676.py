def strip_tail(sequence, values):
    """Strip `values` from the end of `sequence`."""
    return list(reversed(list(strip_head(reversed(sequence), values))))