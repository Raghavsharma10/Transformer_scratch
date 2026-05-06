def _split_chemical_equations(value):
    """
    Split a string with sequential chemical equations into separate strings.

    Each string in the returned iterable represents a single chemical equation
    of the input.
    See the docstrings of `ChemicalEquation` and `ChemicalSystem` for more.

    Parameters
    ----------
    value : `str`
        A string with sequential chemical equations in the mini-language (see
        notes on `ChemicalEquation`).

    Returns
    -------
    iterable of `str`
        An iterable of strings in the format specified by the mini-language
        (see notes on `ChemicalEquation`).

    Examples
    --------
    >>> from pyrrole.core import _split_chemical_equations
    >>> _split_chemical_equations('A + B -> C + D -> D + E <=> F + G <- H + I')
    ['A + B -> C + D', 'C + D -> D + E', 'D + E <=> F + G', 'F + G <- H + I']

    """
    pieces = _split_arrows(value)
    return [(pieces[i] +
             pieces[i + 1] +
             pieces[i + 2]).strip()
            for i in range(0, len(pieces) - 2, 2)]