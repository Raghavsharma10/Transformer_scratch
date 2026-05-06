def toposort(initialAtoms, initialBonds):
    """initialAtoms, initialBonds -> atoms, bonds
    Given the list of atoms and bonds in a ring
    return the topologically sorted atoms and bonds.
    That is each atom is connected to the following atom
    and each bond is connected to the following bond in
    the following manner
    a1 - b1 - a2 - b2 - ... """
    atoms = []
    a_append = atoms.append
    bonds = []
    b_append = bonds.append

    # for the atom and bond hashes
    # we ignore the first atom since we
    # would have deleted it from the hash anyway
    ahash = {}
    bhash = {}
    for atom in initialAtoms[1:]:
        ahash[atom.handle] = 1
        
    for bond in initialBonds:
        bhash[bond.handle] = bond

    next = initialAtoms[0]
    a_append(next)

    # do until all the atoms are gone
    while ahash:
        # traverse to all the connected atoms
        for atom in next.oatoms:
            # both the bond and the atom have to be
            # in our list of atoms and bonds to use
            # ugg, nested if's...  There has to be a
            # better control structure
            if ahash.has_key(atom.handle):
                bond = next.findbond(atom)
                assert bond
                # but wait! the bond has to be in our
                # list of bonds we can use!
                if bhash.has_key(bond.handle):
                    a_append(atom)
                    b_append(bond)
                    del ahash[atom.handle]
                    next = atom
                    break
        else:
            raise RingException("Atoms are not in ring")

    assert len(initialAtoms) == len(atoms)
    assert len(bonds) == len(atoms) - 1
    lastBond = atoms[0].findbond(atoms[-1])
    assert lastBond
    b_append(lastBond)
    return atoms, bonds