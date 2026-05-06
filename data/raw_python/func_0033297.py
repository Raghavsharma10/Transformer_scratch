def _dict_values_sorted_by_key(dictionary):
    # This should be a yield from instead.
    """Internal helper to return the values of a dictionary, sorted by key.
    """
    for _, value in sorted(dictionary.iteritems(), key=operator.itemgetter(0)):
        yield value