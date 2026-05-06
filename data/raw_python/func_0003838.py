def generate_manipulations(
    molecule, bond_stretch_factor=0.15, torsion_amplitude=np.pi,
    bending_amplitude=0.30
):
    """Generate a (complete) set of manipulations

       The result can be used as input for the functions 'randomize_molecule'
       and 'single_random_manipulation'

       Arguments:
         molecule  --  a reference geometry of the molecule, with graph
                       attribute
         bond_stretch_factor  --  the maximum relative change in bond length by
                                  one bond stretch manipulatio
         torsion_amplitude  --  the maximum change a dihdral angle
         bending_amplitude  --  the maximum change in a bending angle

       The return value is a list of RandomManipulation objects. They can be
       used to generate different random distortions of the original molecule.
    """
    do_stretch = (bond_stretch_factor > 0)
    do_double_stretch = (bond_stretch_factor > 0)
    do_bend = (bending_amplitude > 0)
    do_double_bend = (bending_amplitude > 0)
    do_torsion = (torsion_amplitude > 0)

    results = []
    # A) all manipulations that require one bond that cuts the molecule in half
    if do_stretch or do_torsion:
        for affected_atoms1, affected_atoms2, hinge_atoms in iter_halfs_bond(molecule.graph):
            if do_stretch:
                length = np.linalg.norm(
                    molecule.coordinates[hinge_atoms[0]] -
                    molecule.coordinates[hinge_atoms[1]]
                )
                results.append(RandomStretch(
                    affected_atoms1, length*bond_stretch_factor, hinge_atoms
                ))
            if do_torsion and len(affected_atoms1) > 1 and len(affected_atoms2) > 1:
                results.append(RandomTorsion(
                    affected_atoms1, torsion_amplitude, hinge_atoms
                ))
    # B) all manipulations that require a bending angle that cuts the molecule
    #    in two parts
    if do_bend:
        for affected_atoms, hinge_atoms in iter_halfs_bend(molecule.graph):
            results.append(RandomBend(
                affected_atoms, bending_amplitude, hinge_atoms
            ))
    # C) all manipulations that require two bonds that separate two halfs
    if do_double_stretch or do_double_bend:
        for affected_atoms1, affected_atoms2, hinge_atoms in iter_halfs_double(molecule.graph):
            if do_double_stretch:
                length1 = np.linalg.norm(
                    molecule.coordinates[hinge_atoms[0]] -
                    molecule.coordinates[hinge_atoms[1]]
                )
                length2 = np.linalg.norm(
                    molecule.coordinates[hinge_atoms[2]] -
                    molecule.coordinates[hinge_atoms[3]]
                )
                results.append(RandomDoubleStretch(
                    affected_atoms1, 0.5*(length1+length2)*bond_stretch_factor, hinge_atoms
                ))
            if do_double_bend and len(affected_atoms1) > 2 and len(affected_atoms2) > 2:
                if hinge_atoms[0] != hinge_atoms[2]:
                    results.append(RandomTorsion(
                        affected_atoms1, bending_amplitude, (hinge_atoms[0], hinge_atoms[2])
                    ))
                if hinge_atoms[1] != hinge_atoms[3]:
                    results.append(RandomTorsion(
                        affected_atoms2, bending_amplitude, (hinge_atoms[1], hinge_atoms[3])
                    ))
    # Neglect situations where three or more cuts are required.
    return results