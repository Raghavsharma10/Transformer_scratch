def draw(molecule, TraversalType=SmilesTraversal):
    """(molecule)->canonical representation of a molecule
    Well, it's only canonical if the atom symorders are
    canonical, otherwise it's arbitrary.

    atoms must have a symorder attribute
    bonds must have a equiv_class attribute"""
    result = []
    atoms = allAtoms = molecule.atoms

    visitedAtoms = {}
    #
    # Traverse all components of the graph to form
    # the output string
    while atoms:
        atom = _get_lowest_symorder(atoms)
        visitedAtoms[atom] = 1

        visitedBonds = {}
        nextTraverse = TraversalType()
        atomsUsed, bondsUsed = [], []
        _traverse(atom, nextTraverse, None,
                  visitedAtoms, visitedBonds,
                  atomsUsed, bondsUsed, TraversalType)
        atoms = []
        for atom in allAtoms:
            if not visitedAtoms.has_key(atom):
                atoms.append(atom)
        assert nextTraverse.atoms == atomsUsed
        assert nextTraverse.bonds == bondsUsed, "%s %s"%(
            nextTraverse.bonds, bondsUsed)
        

        result.append((str(nextTraverse),
                       atomsUsed, bondsUsed))

    result.sort()
    fragments = []
    for r in result:
        fragments.append(r[0])

    return ".".join(fragments), result