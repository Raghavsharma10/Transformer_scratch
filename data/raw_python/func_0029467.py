def combine_sets(*sets):
    """
    Combine multiple sets to create a single larger set.
    """
    combined = set()
    for s in sets:
        combined.update(s)
    return combined