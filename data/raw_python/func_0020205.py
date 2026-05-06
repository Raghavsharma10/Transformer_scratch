def GetTargetCBVs(model):
    '''
    Returns the design matrix of CBVs for the given target.

    :param model: An instance of the :py:obj:`everest` model for the target

    '''

    # Get the info
    season = model.season
    name = model.name

    # We use the LC light curves as CBVs; there aren't
    # enough SC light curves to get a good set
    if name.endswith('.sc'):
        name = name[:-3]

    model.XCBV = sysrem.GetCBVs(season, model=name,
                                niter=model.cbv_niter,
                                sv_win=model.cbv_win,
                                sv_order=model.cbv_order)