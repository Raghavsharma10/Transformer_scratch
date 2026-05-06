def _get_chemical_equation_piece(species_list, coefficients):
    """
    Produce a string from chemical species and their coefficients.

    Parameters
    ----------
    species_list : iterable of `str`
        Iterable of chemical species.
    coefficients : iterable of `float`
        Nonzero stoichiometric coefficients. The length of `species_list` and
        `coefficients` must be the same. Negative values are made positive and
        zeros are ignored along with their respective species.

    Examples
    --------
    >>> from pyrrole.core import _get_chemical_equation_piece
    >>> _get_chemical_equation_piece(["AcOH"], [2])
    '2 AcOH'
    >>> _get_chemical_equation_piece(["AcO-", "H+"], [-1, -1])
    'AcO- + H+'
    >>> _get_chemical_equation_piece("ABCD", [-2, -1, 0, -1])
    '2 A + B + D'

    """
    def _get_token(species, coefficient):
        if coefficient == 1:
            return '{}'.format(species)
        else:
            return '{:g} {}'.format(coefficient, species)

    bag = []
    for species, coefficient in zip(species_list, coefficients):
        if coefficient < 0:
            coefficient = -coefficient
        if coefficient > 0:
            bag.append(_get_token(species, coefficient))
    return '{}'.format(' + '.join(bag))