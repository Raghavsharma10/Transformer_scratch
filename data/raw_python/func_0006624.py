def _get_nr_prefix(i):
    """
    Helper function for looking up the derived name prefix associated with a Unicode scalar value.

    :param i: Unicode scalar value.
    :return: String with the derived name prefix.
    """
    for lookup_range, prefix_string in _nr_prefix_strings.items():
        if i in lookup_range:
            return prefix_string
    raise ValueError("No prefix string associated with {0}!".format(i))