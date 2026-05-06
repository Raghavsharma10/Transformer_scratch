def pybel_to_json(molecule, name=None):
    """Converts a pybel molecule to json.

    Args:
        molecule: An instance of `pybel.Molecule`
        name: (Optional) If specified, will save a "name" property
    Returns:
       A Python dictionary containing atom and bond data
    """
    # Save atom element type and 3D location.
    atoms = [{'element': table.GetSymbol(atom.atomicnum),
              'location': list(atom.coords)}
             for atom in molecule.atoms]
    # Recover auxiliary data, if exists
    for json_atom, pybel_atom in zip(atoms, molecule.atoms):
        if pybel_atom.partialcharge != 0:
            json_atom['charge'] = pybel_atom.partialcharge
        if pybel_atom.OBAtom.HasData('_atom_site_label'):
            obatom = pybel_atom.OBAtom
            json_atom['label'] = obatom.GetData('_atom_site_label').GetValue()
        if pybel_atom.OBAtom.HasData('color'):
            obatom = pybel_atom.OBAtom
            json_atom['color'] = obatom.GetData('color').GetValue()

    # Save number of bonds and indices of endpoint atoms
    bonds = [{'atoms': [b.GetBeginAtom().GetIndex(),
                        b.GetEndAtom().GetIndex()],
              'order': b.GetBondOrder()}
             for b in ob.OBMolBondIter(molecule.OBMol)]
    output = {'atoms': atoms, 'bonds': bonds, 'units': {}}

    # If there's unit cell data, save it to the json output
    if hasattr(molecule, 'unitcell'):
        uc = molecule.unitcell
        output['unitcell'] = [[v.GetX(), v.GetY(), v.GetZ()]
                              for v in uc.GetCellVectors()]
        density = (sum(atom.atomicmass for atom in molecule.atoms) /
                   (uc.GetCellVolume() * 0.6022))
        output['density'] = density
        output['units']['density'] = 'kg / L'

    # Save the formula to json. Use Hill notation, just to have a standard.
    element_count = Counter(table.GetSymbol(a.atomicnum) for a in molecule)
    hill_count = []
    for element in ['C', 'H']:
        if element in element_count:
            hill_count += [(element, element_count[element])]
            del element_count[element]
    hill_count += sorted(element_count.items())

    # If it's a crystal, then reduce the Hill formula
    div = (reduce(gcd, (c[1] for c in hill_count))
           if hasattr(molecule, 'unitcell') else 1)

    output['formula'] = ''.join(n if c / div == 1 else '%s%d' % (n, c / div)
                                for n, c in hill_count)
    output['molecular_weight'] = molecule.molwt / div
    output['units']['molecular_weight'] = 'g / mol'

    # If the input has been given a name, add that
    if name:
        output['name'] = name

    return output