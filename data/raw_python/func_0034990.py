def canBeAromatic(cycle, pyroleLike):
    """(cycle)-> returns AROMATIC if a ring is conjugatable and
                                  passes the simple tests for aromaticity
                 returns MAYBE if the ring in its present form
                                can be aromatic but is not currently
                         NEVER if the ring can never be aromatic"""
    cycleLength = len(cycle)
    # *******************************************************
    #  check for kekular five membered rings
    if cycleLength == 5:
        # check atom types
        for atom in cycle.atoms:
            if not AROMATIC_PYROLE_ATOMS.has_key(atom.symbol):
                return NEVER

        # do we have exactly one pyrole nitrogen like atom?
        pyroleCount = 0
        for atom in cycle.atoms:
            if pyroleLike.has_key(atom.handle):
                pyrole = atom
                pyroleCount += 1

        
        if pyroleCount < 1 or pyroleCount > 2:
            return NEVER

        # rotate the ring so that we start on the pyrole like atom
        cycle.rotate(pyrole)
        bonds = cycle.bonds[:]
        # check the bonds for a kekular structure
        for index, bond in zip(range(len(bonds)), bonds):
            if bond.bondtype not in AROMATIC_5_RING[index]:
                return MAYBE
            
        return AROMATIC
    # *****************************************************
    #  check for kekular six membered rings
    #  kekular rings must have atoms in the AROMATIC_ATOMS
    #  groups and must belong in 6 membered rings.
    #  bonds must be conjugated
    elif cycleLength == 6:
        # XXX FIX ME -> there is a lot of problems with this
        # code I think, what about bonds that are already fixed?
        for atom in cycle.atoms:
            if not AROMATIC_ATOMS.has_key(atom.symbol):
                return NEVER
            
        bonds = cycle.bonds[:]        
                
        last = None
        switch = {1:2, 2:1}
        while bonds:
            bond = bonds.pop()
            bondtype = bond.bondtype

            if bond.bondorder == 3:
                return NEVER
            
            if last is None:
                if bond.bondtype in [1,2]:
                    last = bond.bondtype
            else:
                if last == 1 and bond.bondtype not in [2,4]:
                    return MAYBE
                elif last == 2 and bond.bondtype not in [1, 4]:
                    return MAYBE

                last = switch[last]
                if bondtype != last:
                    bond.bondorder = last

        return AROMATIC
    
    else:
        # we can never be aromatic
        return NEVER