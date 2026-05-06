def calculate_bin_centers(edges):
    """Calculate the centers of wavelengths bins given their edges.

    Parameters
    ----------
    edges : array-like or `~astropy.units.quantity.Quantity`
        Sequence of bin edges. Must be 1D and have at least two values.
        If not a Quantity, assumed to be in Angstrom.

    Returns
    -------
    centers : `~astropy.units.quantity.Quantity`
        Array of bin centers. Will be 1D, have one less value
        than ``edges``, and also the same unit.

    Raises
    ------
    synphot.exceptions.SynphotError
        Invalid input.

    """
    if not isinstance(edges, u.Quantity):
        edges = edges * u.AA

    if edges.ndim != 1:
        raise exceptions.SynphotError('Bin edges must be 1D array.')

    if edges.size < 2:
        raise exceptions.SynphotError(
            'Bin edges must have at least two values.')

    centers = np.empty(edges.size - 1, dtype=np.float64)
    centers[0] = edges.value[:2].mean()

    for i in range(1, centers.size):
        centers[i] = 2.0 * edges.value[i] - centers[i - 1]

    return centers * edges.unit