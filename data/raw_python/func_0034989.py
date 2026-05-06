def getPyroleLikeAtoms(cycle):
    """cycle->return a dictionary of pyrole nitrogen-like atoms in
    a cycle or a molecule  The dictionary is keyed on the atom.handle"""    
    result = {}
    # the outgoing bonds might need to be single or aromatic
    for atom in cycle.atoms:
        lookup = (atom.symbol, atom.charge, atom.hcount, len(atom.bonds))
        if PyroleTable.get(lookup, 0):
            result[atom.handle] = atom

    return result