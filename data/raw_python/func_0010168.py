def json_to_pybel(data, infer_bonds=False):
    """Converts python data structure to pybel.Molecule.

    This will infer bond data if not specified.

    Args:
        data: The loaded json data of a molecule, as a Python object
        infer_bonds (Optional): If no bonds specified in input, infer them
    Returns:
        An instance of `pybel.Molecule`
    """
    obmol = ob.OBMol()
    obmol.BeginModify()
    for atom in data['atoms']:
        obatom = obmol.NewAtom()
        obatom.SetAtomicNum(table.GetAtomicNum(str(atom['element'])))
        obatom.SetVector(*atom['location'])
        if 'label' in atom:
            pd = ob.OBPairData()
            pd.SetAttribute('_atom_site_label')
            pd.SetValue(atom['label'])
            obatom.CloneData(pd)

    # If there is no bond data, try to infer them
    if 'bonds' not in data or not data['bonds']:
        if infer_bonds:
            obmol.ConnectTheDots()
            obmol.PerceiveBondOrders()
    # Otherwise, use the bonds in the data set
    else:
        for bond in data['bonds']:
            if 'atoms' not in bond:
                continue
            obmol.AddBond(bond['atoms'][0] + 1, bond['atoms'][1] + 1,
                          bond['order'])

    # Check for unit cell data
    if 'unitcell' in data:
        uc = ob.OBUnitCell()
        uc.SetData(*(ob.vector3(*v) for v in data['unitcell']))
        uc.SetSpaceGroup('P1')
        obmol.CloneData(uc)
    obmol.EndModify()

    mol = pybel.Molecule(obmol)

    # Add partial charges
    if 'charge' in data['atoms'][0]:
        mol.OBMol.SetPartialChargesPerceived()
        for atom, pyatom in zip(data['atoms'], mol.atoms):
            pyatom.OBAtom.SetPartialCharge(atom['charge'])

    return mol