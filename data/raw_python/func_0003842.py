def single_random_manipulation(molecule, manipulations, nonbond_thresholds, max_tries=1000):
    """Apply a single random manipulation.

       If no randomized molecule can be generated that survives the nonbond
       check after max_tries repetitions, None is returned. In case of success,
       the randomized molecule and the corresponding transformation is returned.
       The original molecule is not altered.
    """
    for m in range(max_tries):
        random_molecule, transformation = single_random_manipulation_low(molecule, manipulations)
        if check_nonbond(random_molecule, nonbond_thresholds):
            return random_molecule, transformation
    return None