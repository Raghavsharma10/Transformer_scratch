def _prepareSObjects(sObjects):
    '''Prepare a SObject'''
    sObjectsCopy = copy.deepcopy(sObjects)
    if isinstance(sObjectsCopy, dict):
        # If root element is a dict, then this is a single object not an array
        _doPrep(sObjectsCopy)
    else:
        # else this is an array, and each elelment should be prepped.
        for listitems in sObjectsCopy:
            _doPrep(listitems)
    return sObjectsCopy