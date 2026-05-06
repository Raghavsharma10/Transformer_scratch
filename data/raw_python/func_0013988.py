def clean(inst):
    """Routine to return FPMU data cleaned to the specified level

    Parameters
    -----------
    inst : (pysat.Instrument)
        Instrument class object, whose attribute clean_level is used to return
        the desired level of data selectivity.

    Returns
    --------
    Void : (NoneType)
        data in inst is modified in-place.

    Notes
    --------
    No cleaning currently available for FPMU
    """

    inst.data.replace(-999., np.nan, inplace=True) # Te
    inst.data.replace(-9.9999998e+30, np.nan, inplace=True) #Ni

    return None