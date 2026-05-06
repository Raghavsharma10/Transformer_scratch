def merge_by_overlap(sets):
    '''
    Of a list of sets, merge those that overlap, in place.

    The result isn't necessarily a subsequence of the original ``sets``.

    Parameters
    ----------
    sets : ~typing.Sequence[~typing.Set[~typing.Any]]
        Sets of which to merge those that overlap. Empty sets are ignored.

    Notes
    -----
    Implementation is based on `this StackOverflow answer`_. It outperforms all
    other algorithms in the thread (visited at dec 2015) on python3.4 using a
    wide range of inputs.

    .. _this StackOverflow answer: http://stackoverflow.com/a/9453249/1031434

    Examples
    --------
    >>> merge_by_overlap([{1,2}, set(), {2,3}, {4,5,6}, {6,7}])
    [{1,2,3}, {4,5,6,7}]
    '''
    data = sets
    bins = list(range(len(data)))  # Initialize each bin[n] == n
    nums = dict()

    for r, row in enumerate(data):
        if not row:
            data[r] = None
        else:
            for num in row:
                if num not in nums:
                    # New number: tag it with a pointer to this row's bin
                    nums[num] = r
                    continue
                else:
                    dest = _locate_bin(bins, nums[num])
                    if dest == r:
                        continue  # already in the same bin

                    if dest > r:
                        dest, r = r, dest  # always merge into the smallest bin

                    data[dest].update(data[r])
                    data[r] = None
                    # Update our indices to reflect the move
                    bins[r] = dest
                    r = dest 

    # Remove empty bins
    for i in reversed(range(len(data))):
        if not data[i]:
            del data[i]