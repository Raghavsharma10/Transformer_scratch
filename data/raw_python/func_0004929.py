def readbdfv1(filename, bdfext='.bdf', bhfext='.bhf'):
    """Read bdf file (Bessy Data Format v1)

    Input
    -----
    filename: string
        the name of the file

    Output
    ------
    the BDF structure in a dict

    Notes
    -----
    This is an adaptation of the bdf_read.m macro of Sylvio Haas.
    """
    return header.readbhfv1(filename, True, bdfext, bhfext)