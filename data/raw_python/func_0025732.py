def findFirstPar(theDict, name, _depth=0):
    """ Find the given par.  Return tuple: (its own (sub-)dict, its value).
    Returns the first match found, without checking whether the given key name
    is unique or whether it is used in multiple sections. """

    for key in theDict:
        val = theDict[key]
#       print _depth*'   ', key, str(val)[:40]
        if isinstance(val, dict):
            retval = findFirstPar(val, name, _depth=_depth+1) # recurse
            if retval is not None:
                return retval
            # else keep looking
        else:
            if key == name:
                return theDict, theDict[name]
            # else keep looking
    # if we get here then we have searched this whole (sub)-section and its
    # descendants, and found no matches.  only raise if we are at the top.
    if _depth == 0:
        raise KeyError(name)
    else:
        return None