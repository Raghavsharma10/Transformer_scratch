def create_data(*args):
    """
    Produce a single data object from an arbitrary number of different objects.

    This function returns a single `pandas.DataFrame` object from a collection
    of `Atoms` and `pandas.DataFrame` objects. The returned object, already
    indexed by `Atoms.name`, can be promptly used by e.g. `ChemicalSystem`.

    Parameters
    ----------
    *args : `pandas.DataFrame` or `Atoms`-like
        All positional arguments are assumed to be sources of data.

        `Atoms`-like objects (i.e. any object accepted by the `Atoms`
        constructor) become single row records in the final returned
        data object. `pandas.DataFrame` data table objects, on the other hand,
        are concatenated together (by using `pandas.DataFrame.concat`).

    Returns
    -------
    dataframe : `pandas.DataFrame`
        Resulting tabular data object. The returned object is guaranteed to be
        indexed by `Atoms.name`; if no column with this name exists at
        indexing time, a new column (with `None` values) is created for the
        purpose of indexing.

    Notes
    -----
    The returned `pandas.DataFrame` will be indexed by `Atoms.name` (see
    examples below), which might be the same as `Atoms.jobfilename` if no name
    was given to the constructor of `Atoms` (e.g. mapping).

    Examples
    --------
    >>> from pyrrole.atoms import Atoms, create_data, read_cclib
    >>> pyrrole = read_cclib('data/pyrrolate/pyrrole.out', 'pyrrole')
    >>> pyrrolate = read_cclib('data/pyrrolate/pyrrolate.out')
    >>> data = create_data(pyrrole, pyrrolate)
    >>> data['charge']
    name
    pyrrole                         0
    data/pyrrolate/pyrrolate.out   -1
    Name: charge, dtype: int64

    """
    def _prepare_data(data):
        if not isinstance(data, _pd.DataFrame):
            try:
                data = _pd.DataFrame([data.to_series()])
            except AttributeError:
                data = _pd.DataFrame([Atoms(data).to_series()])
        if data.index.name != "name":
            if "name" not in data.columns:
                data["name"] = None
            data = data.set_index("name")
        return data.reset_index()
    args = map(_prepare_data, args)

    dataframe = _pd.concat(args, sort=False)
    return dataframe.set_index("name")