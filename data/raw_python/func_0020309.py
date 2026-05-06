def DetrendFITS(fitsfile, raw=False, season=None, clobber=False, **kwargs):
    """
    De-trend a K2 FITS file using :py:class:`everest.detrender.rPLD`.

    :param str fitsfile: The full path to the FITS file
    :param ndarray aperture: A 2D integer array corresponding to the \
           desired photometric aperture (1 = in aperture, 0 = outside \
           aperture). Default is to interactively select an aperture.
    :param kwargs: Any kwargs accepted by :py:class:`everest.detrender.rPLD`.

    :returns: An :py:class:`everest.Everest` instance.

    """
    # Get info
    EPIC = pyfits.getheader(fitsfile, 0)['KEPLERID']
    if season is None:
        season = pyfits.getheader(fitsfile, 0)['CAMPAIGN']
        if season is None or season == "":
            season = 0
    everestfile = os.path.join(
        everest.missions.k2.TargetDirectory(EPIC, season),
        everest.missions.k2.FITSFile(EPIC, season))

    # De-trend?
    if clobber or not os.path.exists(everestfile):

        # Get raw data
        data = GetData(fitsfile, EPIC, season, clobber=clobber, **kwargs)

        # De-trend
        model = everest.rPLD(EPIC,
                             data=data,
                             season=season, debug=True,
                             clobber=clobber, **kwargs)

        # Publish it
        everest.fits.MakeFITS(model)
        shutil.copyfile(os.path.join(model.dir, model.name + '.pdf'),
                        os.path.join(model.dir,
                                     model._mission.DVSFile(model.ID,
                                                            model.season,
                                                            model.cadence)))

    # Return an Everest instance
    return everest.Everest(EPIC, season=season)