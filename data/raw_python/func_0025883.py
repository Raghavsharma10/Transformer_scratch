def stisExt2PrimKw(stisfiles):
    """
        Several kw which are usually in the primary header
        are in the extension header for STIS. They are copied to
        the primary header for convenience.
        List if kw:
        'DATE-OBS', 'EXPEND', 'EXPSTART', 'EXPTIME'
    """

    kw_list = ['DATE-OBS', 'EXPEND', 'EXPSTART', 'EXPTIME']

    for sfile in stisfiles:
        toclose = False

        if isinstance(sfile, str):
            sfile = fits.open(sfile, mode='update')
            toclose = True

        #d = {}
        for k in kw_list:
            sfile[0].header[k] = sfile[1].header[k]
            sfile[0].header.comments[k] = "Copied from extension header"
        if toclose:
            sfile.close()