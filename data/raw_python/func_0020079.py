def get(ID, pipeline='everest2', campaign=None):
    '''
    Returns the `time` and `flux` for a given EPIC `ID` and
    a given `pipeline` name.

    '''

    log.info('Downloading %s light curve for %d...' % (pipeline, ID))

    # Dev version hack
    if EVEREST_DEV:
        if pipeline.lower() == 'everest2' or pipeline.lower() == 'k2sff':
            from . import Season, TargetDirectory, FITSFile
            if campaign is None:
                campaign = Season(ID)
            fits = os.path.join(TargetDirectory(
                ID, campaign), FITSFile(ID, campaign))
            newdir = os.path.join(KPLR_ROOT, "data", "everest", str(ID))
            if not os.path.exists(newdir):
                os.makedirs(newdir)
            if os.path.exists(fits):
                shutil.copy(fits, newdir)

    if pipeline.lower() == 'everest2':
        s = k2plr.EVEREST(ID, version=2, sci_campaign=campaign)
        time = s.time
        flux = s.flux
    elif pipeline.lower() == 'everest1':
        s = k2plr.EVEREST(ID, version=1, sci_campaign=campaign)
        time = s.time
        flux = s.flux
    elif pipeline.lower() == 'k2sff':
        s = k2plr.K2SFF(ID, sci_campaign=campaign)
        time = s.time
        flux = s.fcor
        # Normalize to the median flux
        s = k2plr.EVEREST(ID, version=2, sci_campaign=campaign)
        flux *= np.nanmedian(s.flux)
    elif pipeline.lower() == 'k2sc':
        s = k2plr.K2SC(ID, sci_campaign=campaign)
        time = s.time
        flux = s.pdcflux
    elif pipeline.lower() == 'raw':
        s = k2plr.EVEREST(ID, version=2, raw=True, sci_campaign=campaign)
        time = s.time
        flux = s.flux
    else:
        raise ValueError('Invalid pipeline: `%s`.' % pipeline)

    return time, flux