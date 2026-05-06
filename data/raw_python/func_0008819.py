def mask_file(regionfile, infile, outfile, negate=False):
    """
    Created a masked version of file, using a region.


    Parameters
    ----------
    regionfile : str
        A file which can be loaded as a :class:`AegeanTools.regions.Region`.
        The image will be masked according to this region.

    infile : str
        Input FITS image.

    outfile : str
        Output FITS image.

    negate :  bool
        If True then pixels *outside* the region are masked.
        Default = False.

    See Also
    --------
    :func:`AegeanTools.MIMAS.mask_plane`
    """
    # Check that the input file is accessible and then open it
    if not os.path.exists(infile): raise AssertionError("Cannot locate fits file {0}".format(infile))
    im = pyfits.open(infile)
    if not os.path.exists(regionfile): raise AssertionError("Cannot locate region file {0}".format(regionfile))
    region = Region.load(regionfile)
    try:
        wcs = pywcs.WCS(im[0].header, naxis=2)
    except:  # TODO: figure out what error is being thrown
        wcs = pywcs.WCS(str(im[0].header), naxis=2)

    if len(im[0].data.shape) > 2:
        data = np.squeeze(im[0].data)
    else:
        data = im[0].data

    print(data.shape)
    if len(data.shape) == 3:
        for plane in range(data.shape[0]):
            mask_plane(data[plane], wcs, region, negate)
    else:
        mask_plane(data, wcs, region, negate)
    im[0].data = data
    im.writeto(outfile, overwrite=True)
    logging.info("Wrote {0}".format(outfile))
    return