def MakeFITS(model, fitsfile=None):
    '''
    Generate a FITS file for a given :py:mod:`everest` run.

    :param model: An :py:mod:`everest` model instance

    '''

    # Get the fits file name
    if fitsfile is None:
        outfile = os.path.join(model.dir, model._mission.FITSFile(
            model.ID, model.season, model.cadence))
    else:
        outfile = os.path.join(model.dir, fitsfile)
    if os.path.exists(outfile) and not model.clobber:
        return
    elif os.path.exists(outfile):
        os.remove(outfile)

    log.info('Generating FITS file...')

    # Create the HDUs
    primary = PrimaryHDU(model)
    lightcurve = LightcurveHDU(model)
    pixels = PixelsHDU(model)
    aperture = ApertureHDU(model)
    images = ImagesHDU(model)
    hires = HiResHDU(model)

    # Combine to get the HDUList
    hdulist = pyfits.HDUList(
        [primary, lightcurve, pixels, aperture, images, hires])

    # Output to the FITS file
    hdulist.writeto(outfile)

    return