def _sortValueIntoGroup(groupKeys, groupLimits, value):
    """ returns the Key of the group a value belongs to
    :param groupKeys: a list/tuple of keys ie ['1-3', '3-5', '5-8', '8-10', '10+']
    :param groupLimits: a list of the limits for the group [1,3,5,8,10,float('inf')] note the first value is an absolute
    minimum and the last an absolute maximum. You can therefore use float('inf')
    :param value:
    :return:
    """

    if not len(groupKeys) == len(groupLimits)-1:
        raise ValueError('len(groupKeys) must equal len(grouplimits)-1 got \nkeys:{0} \nlimits:{1}'.format(groupKeys,
                                                                                                         groupLimits))

    if math.isnan(value):
        return 'Uncertain'

    # TODO add to other if bad value or outside limits
    keyIndex = None

    if value == groupLimits[0]:  # if value is == minimum skip the comparison
        keyIndex = 1
    elif value == groupLimits[-1]:  # if value is == minimum skip the comparison
        keyIndex = len(groupLimits)-1
    else:
        for i, limit in enumerate(groupLimits):
            if value < limit:
                keyIndex = i
                break

    if keyIndex == 0:  # below the minimum
        raise BelowLimitsError('Value {0} below limit {1}'.format(value, groupLimits[0]))

    if keyIndex is None:
        raise AboveLimitsError('Value {0} above limit {1}'.format(value, groupLimits[-1]))

    return groupKeys[keyIndex-1]