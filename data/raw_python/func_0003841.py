def randomize_molecule_low(molecule, manipulations):
    """Return a randomized copy of the molecule, without the nonbond check."""

    manipulations = copy.copy(manipulations)
    shuffle(manipulations)
    coordinates = molecule.coordinates.copy()
    for manipulation in manipulations:
        manipulation.apply(coordinates)
    return molecule.copy_with(coordinates=coordinates)