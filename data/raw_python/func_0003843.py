def single_random_manipulation_low(molecule, manipulations):
    """Return a randomized copy of the molecule, without the nonbond check."""

    manipulation = sample(manipulations, 1)[0]
    coordinates = molecule.coordinates.copy()
    transformation = manipulation.apply(coordinates)
    return molecule.copy_with(coordinates=coordinates), transformation