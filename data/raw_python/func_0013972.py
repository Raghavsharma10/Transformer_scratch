def remove_icon_names(inst, target=None):
    """Removes leading text on ICON project variable names

    Parameters
    ----------
    inst : pysat.Instrument
        ICON associated pysat.Instrument object
    target : str
        Leading string to remove. If none supplied,
        ICON project standards are used to identify and remove
        leading text

    Returns
    -------
    None
        Modifies Instrument object in place


    """
  
    if target is None:
        lev = inst.tag
        if lev == 'level_2':
            lev = 'L2'
        elif lev == 'level_0':
            lev = 'L0'
        elif lev == 'level_0p':
            lev = 'L0P'
        elif lev == 'level_1.5':
            lev = 'L1-5'
        elif lev == 'level_1':
            lev = 'L1'
        else:
            raise ValueError('Uknown ICON data level')
        
        # get instrument code
        sid = inst.sat_id.lower()
        if sid == 'a':
            sid = 'IVM_A'
        elif sid == 'b':
            sid = 'IVM_B'
        else:
            raise ValueError('Unknown ICON satellite ID')
        prepend_str = '_'.join(('ICON', lev, sid)) + '_'
    else:
        prepend_str = target

    inst.data.rename(columns=lambda x: x.split(prepend_str)[-1], inplace=True)
    inst.meta.data.rename(index=lambda x: x.split(prepend_str)[-1], inplace=True)
    orig_keys = inst.meta.keys_nD()  
    for key in orig_keys:
        new_key = key.split(prepend_str)[-1]
        new_meta = inst.meta.pop(key)
        new_meta.data.rename(index=lambda x: x.split(prepend_str)[-1], inplace=True)
        inst.meta[new_key] = new_meta
        
    return