def remove_lower_overlapping(current, higher):
    """
    >>> remove_lower_overlapping([], [('a', 0, 5)])
    [('a', 0, 5)]
    >>> remove_lower_overlapping([('z', 0, 4)], [('a', 0, 5)])
    [('a', 0, 5)]
    >>> remove_lower_overlapping([('z', 5, 6)], [('a', 0, 5)])
    [('z', 5, 6), ('a', 0, 5)]
    """
    for (match, h_start, h_end) in higher:
        overlaps = list(overlapping_at(h_start, h_end, current))
        for overlap in overlaps:
            del current[overlap]
        if len(overlaps) > 0:
            # Keeps order in place
            current.insert(overlaps[0], (match, h_start, h_end))
        else:
            current.append((match, h_start, h_end))

    return current