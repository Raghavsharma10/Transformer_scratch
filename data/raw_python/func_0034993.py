def aromatize(molecule, usedPyroles=None):    
    """(molecule, usedPyroles=None)->aromatize a molecular graph
    usedPyroles is a dictionary that holds the pyrole like
    atoms that are used in the conversion process.
    The following valence checker may need this information"""
    
    pyroleLike = getPyroleLikeAtoms(molecule)

    if usedPyroles is None:
        usedPyroles = {}
        
    cyclesToCheck = []
    # determine which cycles came in marked as aromatic
    # and which need to be checked form the kekular form
    #
    # if a cycle came in as aromatic, convert it
    # before going on.
    for cycle in molecule.cycles:
        for atom in cycle.atoms:
            if not atom.aromatic:
                cyclesToCheck.append(cycle)
                break
        else:
            if not convert(cycle, pyroleLike, usedPyroles):
                # XXX FIX ME
                # oops, an aromatic ring came in but
                # we can't convert it.  This is an error
                # daylight would conjugate the ring
                raise PinkyError("Bad initial aromaticity")

    # keep checking rings until something happens
    while 1:
        # assume nothing happened
        needToCheckAgain = 0

        _cyclesToCheck = []
        for cycle in cyclesToCheck:
            canAromatic = canBeAromatic(cycle, pyroleLike)
            if canAromatic == NEVER:
                # the ring can NEVER EVER be aromatic, so remove it for good
                pass
            elif canAromatic and convert(cycle, pyroleLike, usedPyroles):
                needToCheckAgain = 1
            else:
                _cyclesToCheck.append(cycle)

        cyclesToCheck = _cyclesToCheck
        if not needToCheckAgain:
            break

    # fix bonds that have no bondorder if necessary
    molecule = fixBonds(molecule, pyroleLike)
    # add implicit hydrogens
    return addHydrogens(molecule, usedPyroles)