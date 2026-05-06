def read_cclib(value, name=None):
    """
    Create an `Atoms` object from data attributes parsed by cclib.

    `cclib <https://cclib.github.io/>`_ is an open source library, written in
    Python, for parsing and interpreting the results (logfiles) of
    computational chemistry packages.

    Parameters
    ----------
    value : `str`, `cclib.parser.logfileparser.Logfile`, `cclib.parser.data.ccData`
        A path to a logfile, or either a cclib job object (i.e., from
        `cclib.ccopen`), or cclib data object (i.e., from ``job.parse()``).
    name : `str`, optional
        Name for chemical species. If not given, this is set to the logfile
        path, if known. Chemical equations mention this name when refering to
        the returned object.

    Returns
    -------
    molecule : `Atoms`
        All attributes obtainable by cclib are made available as attributes in
        the returned object.

    Examples
    --------
    >>> from pyrrole.atoms import read_cclib
    >>> molecule = read_cclib('data/pyrrolate/pyrrole.out')
    >>> molecule.atomnos
    array([6, 6, 6, 6, 7, 1, 1, 1, 1, 1], dtype=int32)
    >>> molecule.charge
    0

    """
    if isinstance(value, _logfileparser.Logfile):
        # TODO: test this case.
        jobfilename = value.filename
        ccdata = value.parse()
    elif isinstance(value, _data.ccData):
        # TODO: test this case.
        jobfilename = None
        ccdata = value
    else:
        # TODO: test this case.
        ccobj = _cclib.ccopen(value)
        jobfilename = ccobj.filename
        ccdata = ccobj.parse()

    if name is None:
        name = jobfilename

    attributes = ccdata.getattributes()
    attributes.update({
        'name': name,
        'jobfilename': jobfilename,
    })

    return Atoms(attributes)