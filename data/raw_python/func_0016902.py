def source_range(start, end, nr_var_dict):
    """
    Given a range of source numbers, as well as a dictionary
    containing the numbers of each source, returns a dictionary
    containing tuples of the start and end index
    for each source variable type.
    """

    return OrderedDict((k, e-s)
        for k, (s, e)
        in source_range_tuple(start, end, nr_var_dict).iteritems())