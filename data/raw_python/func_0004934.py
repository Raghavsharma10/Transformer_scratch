def readbdfv2(filename, bdfext='.bdf', bhfext='.bhf'):
    """Read a version 2 Bessy Data File

    Inputs
    ------
    filename: string
        the name of the input file. One can give the complete header or datafile
        name or just the base name without the extensions.
    bdfext: string, optional
        the extension of the data file
    bhfext: string, optional
        the extension of the header file

    Output
    ------
    the data structure in a dict. Header is loaded implicitely.

    Notes
    -----
    BDFv2 header and scattering data are stored separately in the header and
    the data files. Given the file name both are loaded.
    """
    datas = header.readbhfv2(filename, True, bdfext, bhfext)
    return datas