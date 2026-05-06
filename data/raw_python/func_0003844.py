def random_dimer(molecule0, molecule1, thresholds, shoot_max):
    """Create a random dimer.

       molecule0 and molecule1 are placed in one reference frame at random
       relative positions. Interatomic distances are above the thresholds.
       Initially a dimer is created where one interatomic distance approximates
       the threshold value. Then the molecules are given an additional
       separation in the range [0, shoot_max].

       thresholds has the following format:
       {frozenset([atom_number1, atom_number2]): distance}
    """

    # apply a random rotation to molecule1
    center = np.zeros(3, float)
    angle = np.random.uniform(0, 2*np.pi)
    axis = random_unit()
    rotation = Complete.about_axis(center, angle, axis)
    cor1 = np.dot(molecule1.coordinates, rotation.r)

    # select a random atom in each molecule
    atom0 = np.random.randint(len(molecule0.numbers))
    atom1 = np.random.randint(len(molecule1.numbers))

    # define a translation of molecule1 that brings both atoms in overlap
    delta = molecule0.coordinates[atom0] - cor1[atom1]
    cor1 += delta

    # define a random direction
    direction = random_unit()
    cor1 += 1*direction

    # move molecule1 along this direction until all intermolecular atomic
    # distances are above the threshold values
    threshold_mat = np.zeros((len(molecule0.numbers), len(molecule1.numbers)), float)
    distance_mat = np.zeros((len(molecule0.numbers), len(molecule1.numbers)), float)
    for i1, n1 in enumerate(molecule0.numbers):
        for i2, n2 in enumerate(molecule1.numbers):
            threshold = thresholds.get(frozenset([n1, n2]))
            threshold_mat[i1, i2] = threshold**2
    while True:
        cor1 += 0.1*direction
        distance_mat[:] = 0
        for i in 0, 1, 2:
            distance_mat += np.subtract.outer(molecule0.coordinates[:, i], cor1[:, i])**2
        if (distance_mat > threshold_mat).all():
            break

    # translate over a random distance [0, shoot] along the same direction
    # (if necessary repeat until no overlap is found)
    while True:
        cor1 += direction*np.random.uniform(0, shoot_max)
        distance_mat[:] = 0
        for i in 0, 1, 2:
            distance_mat += np.subtract.outer(molecule0.coordinates[:, i], cor1[:, i])**2
        if (distance_mat > threshold_mat).all():
            break

    # done
    dimer = Molecule(
        np.concatenate([molecule0.numbers, molecule1.numbers]),
        np.concatenate([molecule0.coordinates, cor1])
    )
    dimer.direction = direction
    dimer.atom0 = atom0
    dimer.atom1 = atom1
    return dimer