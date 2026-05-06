def to_rdkit_molecule(data):
    """
    MoleculeContainer to RDKit molecule object converter
    """
    mol = RWMol()
    conf = Conformer()
    mapping = {}
    is_3d = False
    for n, a in data.atoms():
        ra = Atom(a.number)
        ra.SetAtomMapNum(n)
        if a.charge:
            ra.SetFormalCharge(a.charge)
        if a.isotope != a.common_isotope:
            ra.SetIsotope(a.isotope)
        if a.radical:
            ra.SetNumRadicalElectrons(a.radical)
        mapping[n] = m = mol.AddAtom(ra)
        conf.SetAtomPosition(m, (a.x, a.y, a.z))
        if a.z:
            is_3d = True
    if not is_3d:
        conf.Set3D(False)

    for n, m, b in data.bonds():
        mol.AddBond(mapping[n], mapping[m], _bond_map[b.order])

    mol.AddConformer(conf)
    SanitizeMol(mol)
    return mol