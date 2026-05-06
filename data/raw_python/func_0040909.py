def subbrick(dset,label,coef=False,tstat=False,fstat=False,rstat=False,number_only=False):
    ''' returns a string referencing the given subbrick within a dset

    This method reads the header of the dataset ``dset``, finds the subbrick whose
    label matches ``label`` and returns a string of type ``dataset[X]``, which can
    be used by most AFNI programs to refer to a subbrick within a file

    The options coef, tstat, fstat, and rstat will add the suffix that is
    appended to the label by 3dDeconvolve

    :coef:  "#0_Coef"
    :tstat: "#0_Tstat"
    :fstat: "_Fstat"
    :rstat: "_R^2"

    If ``coef`` or ``tstat`` are set to a number, it will use that parameter number
    (instead of 0), for models that use multiple parameters (e.g., "TENT").

    if ``number_only`` is set to ``True``, will only return the subbrick number instead of a string
    '''

    if coef is not False:
        if coef is True:
            coef = 0
        label += "#%d_Coef" % coef
    elif tstat != False:
        if tstat==True:
            tstat = 0
        label += "#%d_Tstat" % tstat
    elif fstat:
        label += "_Fstat"
    elif rstat:
        label += "_R^2"

    info = nl.dset_info(dset)
    if info==None:
        nl.notify('Error: Couldn\'t get info from dset "%s"'%dset,level=nl.level.error)
        return None
    i = info.subbrick_labeled(label)
    if number_only:
        return i
    return '%s[%d]' % (dset,i)