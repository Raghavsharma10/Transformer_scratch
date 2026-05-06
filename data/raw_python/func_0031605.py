def _check_data(data):
    """
    Check a data object for inconsistencies.

    Parameters
    ----------
    data : `pandas.DataFrame`
        A `data` object, i.e., a table whose rows store information about
        chemical species, indexed by chemical species.

    Warns
    -----
    UserWarning
        Warned if a ground state species has one or more imaginary vibrational
        frequencies, or if a transition state species has zero, two or more
        imaginary vibrational frequencies.

    Examples
    --------
    >>> import pandas as pd
    >>> from pyrrole.core import _check_data
    >>> data = (pd.DataFrame([{'name': 'A', 'vibfreqs': [0., 1., 2.]},
    ...                       {'name': 'B', 'vibfreqs': [0., -1., 2.]},
    ...                       {'name': 'C', 'vibfreqs': [0., -1., -2.]},
    ...                       {'name': 'A#', 'vibfreqs': [0., 1., 2.]},
    ...                       {'name': 'C#', 'vibfreqs': [0., -2., -1.]},
    ...                       {'name': 'B#', 'vibfreqs': [0., -1., 2.]}])
    ...         .set_index('name'))
    >>> _check_data(data)

    """
    if "vibfreqs" in data.columns:
        for species in data.index:
            vibfreqs = data.loc[species, "vibfreqs"]
            nimagvibfreqs = _np.sum(_np.array(vibfreqs) < 0)
            if species[-1] == '#' and nimagvibfreqs != 1:
                _warnings.warn("'{}' should have 1 imaginary vibfreqs but {} "
                               "found".format(species, nimagvibfreqs))
            elif species[-1] != '#' and nimagvibfreqs != 0:
                _warnings.warn("'{}' should have no imaginary vibfreqs but {} "
                               "found".format(species, nimagvibfreqs))