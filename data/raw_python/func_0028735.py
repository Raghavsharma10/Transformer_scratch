def save(sources, targets, masked=False):
    """
    Save the numeric results of each source into its corresponding target.

    Parameters
    ----------
    sources: list
        The list of source arrays for saving from; limited to length 1.
    targets: list
        The list of target arrays for saving to; limited to length 1.
    masked: boolean
        Uses a masked array from sources if True.

    """
    # TODO: Remove restriction
    assert len(sources) == 1 and len(targets) == 1
    array = sources[0]
    target = targets[0]

    # Request bitesize pieces of the source and assign them to the
    # target.
    # NB. This algorithm does not use the minimal number of chunks.
    #   e.g. If the second dimension could be sliced as 0:99, 99:100
    #       then clearly the first dimension would have to be single
    #       slices for the 0:99 case, but could be bigger slices for the
    #       99:100 case.
    # It's not yet clear if this really matters.
    all_slices = _all_slices(array)
    for index in np.ndindex(*[len(slices) for slices in all_slices]):
        keys = tuple(slices[i] for slices, i in zip(all_slices, index))
        if masked:
            target[keys] = array[keys].masked_array()
        else:
            target[keys] = array[keys].ndarray()