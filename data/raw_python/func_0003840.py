def randomize_molecule(molecule, manipulations, nonbond_thresholds, max_tries=1000):
    """Return a randomized copy of the molecule.

       If no randomized molecule can be generated that survives the nonbond
       check after max_tries repetitions, None is returned. In case of success,
       the randomized molecule is returned. The original molecule is not
       altered.
    """
    for m in range(max_tries):
        random_molecule = randomize_molecule_low(molecule, manipulations)
        if check_nonbond(random_molecule, nonbond_thresholds):
            return random_molecule