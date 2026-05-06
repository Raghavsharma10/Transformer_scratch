def check_oscntab(oscntab, ccdamp, xsize, ysize, leading, trailing):
    """Check if the supplied parameters are in the
    ``OSCNTAB`` reference file.

    .. note:: Even if an entry does not exist in ``OSCNTAB``,
              as long as the subarray does not have any overscan,
              it should not be a problem for CALACS.

    .. note:: This function does not check the virtual bias rows.

    Parameters
    ----------
    oscntab : str
        Path to the ``OSCNTAB`` reference file being checked against.

    ccdamp : str
        Amplifier(s) used to read out the CCDs.

    xsize : int
        Number of columns in the readout.

    ysize : int
        Number of rows in the readout.

    leading : int
        Number of columns in the bias section ("TRIMX1" to be trimmed off
        by ``BLEVCORR``) on the A/C amplifiers side of the CCDs.

    trailing : int
        Number of columns in the bias section ("TRIMX2" to be trimmed off
        by ``BLEVCORR``) on the B/D amplifiers side of the CCDs.

    Returns
    -------
    supported : bool
        Result of test if input parameters are in ``OSCNTAB``.

    """
    tab = Table.read(oscntab)
    ccdamp = ccdamp.lower().rstrip()
    for row in tab:
        if (row['CCDAMP'].lower().rstrip() in ccdamp and
                row['NX'] == xsize and row['NY'] == ysize and
                row['TRIMX1'] == leading and row['TRIMX2'] == trailing):
            return True
    return False