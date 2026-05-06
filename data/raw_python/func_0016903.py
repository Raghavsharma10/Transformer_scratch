def source_range_slices(start, end, nr_var_dict):
    """
    Given a range of source numbers, as well as a dictionary
    containing the numbers of each source, returns a dictionary
    containing slices for each source variable type.
    """

    return OrderedDict((k, slice(s,e,1))
        for k, (s, e)
        in source_range_tuple(start, end, nr_var_dict).iteritems())