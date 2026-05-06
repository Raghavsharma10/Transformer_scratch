def invariants(mol):
    """Generate initial atom identifiers using atomic invariants"""
    atom_ids = {}
    for a in mol.atoms:
        components = []
        components.append(a.number)
        components.append(len(a.oatoms))
        components.append(a.hcount)
        components.append(a.charge)
        components.append(a.mass)
        if len(a.rings) > 0:
            components.append(1)

        atom_ids[a.index] = gen_hash(components)

    return atom_ids