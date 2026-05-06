def source_range_tuple(start, end, nr_var_dict):
    """
    Given a range of source numbers, as well as a dictionary
    containing the numbers of each source, returns a dictionary
    containing tuples of the start and end index
    for each source variable type.
    """

    starts = np.array([0 for nr_var in SOURCE_VAR_TYPES.itervalues()])
    ends = np.array([nr_var_dict[nr_var] if nr_var in nr_var_dict else 0
        for nr_var in SOURCE_VAR_TYPES.itervalues()])
    sum_counts = np.cumsum(ends)
    idx = np.arange(len(starts))

    # Find the intervals containing the
    # start and ending indices
    start_idx, end_idx = np.searchsorted(
        sum_counts, [start, end], side='right')

    # Handle edge cases
    if end >= sum_counts[-1]:
        end = sum_counts[-1]
        end_idx = len(sum_counts) - 1

    # Find out which variable counts fall within the range
    # of the supplied indices and zero those outside this range
    invalid = np.logical_not(np.logical_and(start_idx <= idx, idx <= end_idx))
    starts[invalid] = ends[invalid] = 0

    # Modify the associated starting and ending positions
    starts[start_idx] = start
    ends[end_idx] = end

    if start >= sum_counts[0]:
        starts[start_idx] -= sum_counts[start_idx-1]

    if end >= sum_counts[0]:
        ends[end_idx] -= sum_counts[end_idx-1]

    return OrderedDict((n, (starts[i], ends[i]))
        for i, n in enumerate(SOURCE_VAR_TYPES.values()))