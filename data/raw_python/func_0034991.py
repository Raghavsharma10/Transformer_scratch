def convert(cycle, pyroleLike, usedPyroles):
    """cycle, pyroleLike, aromatic=0-> aromatize the cycle
    pyroleLike is a lookup of the pyrole like atoms in the
    cycle.
    return 1 if the cycle was aromatized
           2 if the cycle could not be aromatized"""

    bonds = cycle.bonds
    atoms = cycle.atoms
    initialBondStates = []
    initialAtomStates = []
    _usedPyroles = {}
    for bond in bonds:
        # store the initial states but assume the
        # bond is aromatic
        initialBondStates.append((bond, bond.symbol,
                                  bond.bondorder, bond.bondtype,
                                  bond.aromatic, bond.stereo))
        # XXX FIX ME
        # until we get proper conjugation, aromatic bond orders
        # are 1.5
        bond.reset(':', bond.bondorder, 4, bond.fixed, bond.stereo)
        
    aromatized = 1
    for atom in atoms:
        initialAtomStates.append((atom, atom.aromatic))
        atom.aromatic = 1

        nonhydrogens = atom.sumBondOrders() + atom.charge

        # look for the lowest valence where we don't
        #  have to change the charge of the atom to
        #  fill the valences
        for valence in atom.valences:
            neededHydrogens = int(valence - nonhydrogens)
            if neededHydrogens >= 0:
                break
        else:
            # we can't change the aromaticity and have correct
            # valence.
            #
            # there is one special case of a five membered
            #  ring and a pyrole nitrogen like atom we need
            #  to look for.
            if len(cycle) == 5 and pyroleLike.has_key(atom.handle):
                _usedPyroles[atom.handle] = 1
            else:
                # nope, the valences don't work out so
                # we can't aromatize
                aromatized = 0
                break

    # sanity check, this should be true because of the
    # canBeAromatic routine above
    assert len(_usedPyroles) <=1, "Too many used pyroles!"
    
    cycle.aromatic = aromatized
    if not aromatized:
        for bond, symbol, order, bondtype, aromatic, stereo in initialBondStates:
            bond.reset(symbol, order, bondtype, bond.fixed, stereo)

        for atom, aromatic in initialAtomStates:
            atom.aromatic = aromatic
    else:
        # we used some pyroles, we'll have to send these to
        # the valence checker later
        usedPyroles.update(_usedPyroles)

    return aromatized