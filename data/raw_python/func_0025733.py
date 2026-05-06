def findScopedPar(theDict, scope, name):
    """ Find the given par.  Return tuple: (its own (sub-)dict, its value). """
    # Do not search (like findFirstPar), but go right to the correct
    # sub-section, and pick it up.  Assume it is there as stated.
    if len(scope):
        theDict = theDict[scope] # ! only goes one level deep - enhance !
    return theDict, theDict[name]