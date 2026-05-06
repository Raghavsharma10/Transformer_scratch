def clean(inst):
    """Routine to return VEFI data cleaned to the specified level

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
    'dusty' or 'clean' removes data when interpolation flag is set to 1
    """
    
    if (inst.clean_level == 'dusty') | (inst.clean_level == 'clean'):
        idx, = np.where(inst['B_flag'] == 0)
        inst.data = inst[idx, :]

    return None