def from_rdkit_molecule(data):
    """
    RDKit molecule object to MoleculeContainer converter
    """
    m = MoleculeContainer()
    atoms, mapping = [], []
    for a in data.GetAtoms():
        atom = {'element': a.GetSymbol(), 'charge': a.GetFormalCharge()}
        atoms.append(atom)
        mapping.append(a.GetAtomMapNum())

        isotope = a.GetIsotope()
        if isotope:
            atom['isotope'] = isotope
        radical = a.GetNumRadicalElectrons()
        if radical:
            atom['multiplicity'] = radical + 1

    conformers = data.GetConformers()
    if conformers:
        for atom, (x, y, z) in zip(atoms, conformers[0].GetPositions()):
            atom['x'] = x
            atom['y'] = y
            atom['z'] = z

    for atom, mapping in zip(atoms, mapping):
        a = m.add_atom(atom)
        if mapping:
            m.atom(a)._parsed_mapping = mapping

    for bond in data.GetBonds():
        m.add_bond(bond.GetBeginAtomIdx() + 1, bond.GetEndAtomIdx() + 1, _rdkit_bond_map[bond.GetBondType()])

    return m