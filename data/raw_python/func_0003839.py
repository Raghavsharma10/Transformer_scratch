def check_nonbond(molecule, thresholds):
    """Check whether all nonbonded atoms are well separated.

       If a nonbond atom pair is found that has an interatomic distance below
       the given thresholds. The thresholds dictionary has the following format:
       {frozenset([atom_number1, atom_number2]): distance}

       When random geometries are generated for sampling the conformational
       space of a molecule without strong repulsive nonbonding interactions, try
       to underestimate the thresholds at first instance and exclude bond
       stretches and bending motions for the random manipuulations. Then compute
       the forces projected on the nonbonding distance gradients. The distance
       for which the absolute value of these gradients drops below 100 kJ/mol is
       a coarse guess of a proper threshold value.
    """

    # check that no atoms overlap
    for atom1 in range(molecule.graph.num_vertices):
        for atom2 in range(atom1):
            if molecule.graph.distances[atom1, atom2] > 2:
                distance = np.linalg.norm(molecule.coordinates[atom1] - molecule.coordinates[atom2])
                if distance < thresholds[frozenset([molecule.numbers[atom1], molecule.numbers[atom2]])]:
                    return False
    return True