def countKey(theDict, name):
    """ Return the number of times the given par exists in this dict-tree,
    since the same key name may be used in different sections/sub-sections. """

    retval = 0
    for key in theDict:
        val = theDict[key]
        if isinstance(val, dict):
            retval += countKey(val, name) # recurse
        else:
            if key == name:
                retval += 1
                # can't break, even tho we found a hit, other items on
                # this level will not be named "name", but child dicts
                # may have further counts
    return retval