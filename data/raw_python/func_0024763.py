def calculate_bin_widths(edges):
    """Calculate the widths of wavelengths bins given their edges.

    Parameters
    ----------
    edges : array-like or `~astropy.units.quantity.Quantity`
        Sequence of bin edges. Must be 1D and have at least two values.
        If not a Quantity, assumed to be in Angstrom.

    Returns
    -------
    widths : `~astropy.units.quantity.Quantity`
        Array of bin widths. Will be 1D, have one less value
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

    return np.abs(edges[1:] - edges[:-1])