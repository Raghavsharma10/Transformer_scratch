def setup_hydrocarbon_ff(graph):
    """Create a simple ForceField object for hydrocarbons based on the graph."""
    # A) Define parameters.
    # the bond parameters:
    bond_params = {
        (6, 1): 310*kcalmol/angstrom**2,
        (6, 6): 220*kcalmol/angstrom**2,
    }
    # for every (a, b), also add (b, a)
    for key, val in list(bond_params.items()):
        if key[0] != key[1]:
            bond_params[(key[1], key[0])] = val
    # the bend parameters
    bend_params = {
        (1, 6, 1): 35*kcalmol/rad**2,
        (1, 6, 6): 30*kcalmol/rad**2,
        (6, 6, 6): 60*kcalmol/rad**2,
    }
    # for every (a, b, c), also add (c, b, a)
    for key, val in list(bend_params.items()):
        if key[0] != key[2]:
            bend_params[(key[2], key[1], key[0])] = val

    # B) detect all internal coordinates and corresponding energy terms.
    terms = []
    # bonds
    for i0, i1 in graph.edges:
        K = bond_params[(graph.numbers[i0], graph.numbers[i1])]
        terms.append(BondStretchTerm(K, i0, i1))
    # bends (see b_bending_angles.py for the explanation)
    for i1 in range(graph.num_vertices):
        n = list(graph.neighbors[i1])
        for index, i0 in enumerate(n):
            for i2 in n[:index]:
                K = bend_params[(graph.numbers[i0], graph.numbers[i1], graph.numbers[i2])]
                terms.append(BendAngleTerm(K, i0, i1, i2))

    # C) Create and return the force field
    return ForceField(terms)