def _dump_cml_molecule(f, molecule):
    """Dump a single molecule to a CML file

       Arguments:
        | ``f``  --  a file-like object
        | ``molecule``  --  a Molecule instance
    """
    extra = getattr(molecule, "extra", {})
    attr_str = " ".join("%s='%s'" % (key, value) for key, value in extra.items())
    f.write(" <molecule id='%s' %s>\n" % (molecule.title, attr_str))
    f.write("  <atomArray>\n")
    atoms_extra = getattr(molecule, "atoms_extra", {})
    for counter, number, coordinate in zip(range(molecule.size), molecule.numbers, molecule.coordinates/angstrom):
        atom_extra = atoms_extra.get(counter, {})
        attr_str = " ".join("%s='%s'" % (key, value) for key, value in atom_extra.items())
        f.write("   <atom id='a%i' elementType='%s' x3='%s' y3='%s' z3='%s' %s />\n" % (
            counter, periodic[number].symbol, coordinate[0],  coordinate[1],
            coordinate[2], attr_str,
        ))
    f.write("  </atomArray>\n")
    if molecule.graph is not None:
        bonds_extra = getattr(molecule, "bonds_extra", {})
        f.write("  <bondArray>\n")
        for edge in molecule.graph.edges:
            bond_extra = bonds_extra.get(edge, {})
            attr_str = " ".join("%s='%s'" % (key, value) for key, value in bond_extra.items())
            i1, i2 = edge
            f.write("   <bond atomRefs2='a%i a%i' %s />\n" % (i1, i2, attr_str))
        f.write("  </bondArray>\n")
    f.write(" </molecule>\n")