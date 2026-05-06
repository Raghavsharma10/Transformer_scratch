def calculate_bin_edges(centers):
    """Calculate the edges of wavelength bins given the centers.

    The algorithm calculates bin edges as the midpoints between bin centers
    and treats the first and last bins as symmetric about their centers.

    Parameters
    ----------
    centers : array-like or `~astropy.units.quantity.Quantity`
        Sequence of bin centers. Must be 1D and have at least two values.
        If not a Quantity, assumed to be in Angstrom.

    Returns
    -------
    edges : `~astropy.units.quantity.Quantity`
        Array of bin edges. Will be 1D, have one more value
        than ``centers``, and also the same unit.

    Raises
    ------
    synphot.exceptions.SynphotError
        Invalid input.

    """
    if not isinstance(centers, u.Quantity):
        centers = centers * u.AA

    if centers.ndim != 1:
        raise exceptions.SynphotError('Bin centers must be 1D array.')

    if centers.size < 2:
        raise exceptions.SynphotError(
            'Bin centers must have at least two values.')

    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = (centers.value[1:] + centers.value[:-1]) * 0.5

    # Compute the first and last by making them symmetric
    edges[0] = 2.0 * centers.value[0] - edges[1]
    edges[-1] = 2.0 * centers.value[-1] - edges[-2]

    return edges * centers.unit