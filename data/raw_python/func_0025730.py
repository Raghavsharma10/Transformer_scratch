def flattenDictTree(aDict):
    """ Takes a dict of vals and dicts (so, a tree) as input, and returns
    a flat dict (only one level) as output.  All key-vals are moved to
    the top level.  Sub-section dict names (keys) are ignored/dropped.
    If there are name collisions, an error is raised. """
    retval = {}
    for k in aDict:
        val = aDict[k]
        if isinstance(val, dict):
            # This val is a dict, get its data (recursively) into a flat dict
            subDict = flattenDictTree(val)
            # Merge its dict of data into ours, watching for NO collisions
            rvKeySet  = set(retval.keys())
            sdKeySet = set(subDict.keys())
            intr = rvKeySet.intersection(sdKeySet)
            if len(intr) > 0:
                raise DuplicateKeyError("Flattened dict already has "+ \
                    "key(s): "+str(list(intr))+" - cannot flatten this.")

            else:
                retval.update(subDict)
        else:
            if k in retval:
                raise DuplicateKeyError("Flattened dict already has key: "+\
                                        k+" - cannot flatten this.")
            else:
                retval[k] = val
    return retval