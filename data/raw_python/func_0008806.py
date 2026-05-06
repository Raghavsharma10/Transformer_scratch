def filter_image(im_name, out_base, step_size=None, box_size=None, twopass=False, cores=None, mask=True, compressed=False, nslice=None):
    """
    Create a background and noise image from an input image.
    Resulting images are written to `outbase_bkg.fits` and `outbase_rms.fits`

    Parameters
    ----------
    im_name : str or HDUList
        Image to filter. Either a string filename or an astropy.io.fits.HDUList.
    out_base : str
        The output filename base. Will be modified to make _bkg and _rms files.
    step_size : (int,int)
        Tuple of the x,y step size in pixels
    box_size : (int,int)
        The size of the box in piexls
    twopass : bool
        Perform a second pass calculation to ensure that the noise is not contaminated by the background.
        Default = False
    cores : int
        Number of CPU corse to use.
        Default = all available
    nslice : int
        The image will be divided into this many horizontal stripes for processing.
        Default = None = equal to cores
    mask : bool
        Mask the output array to contain np.nna wherever the input array is nan or not finite.
        Default = true
    compressed : bool
        Return a compressed version of the background/noise images.
        Default = False

    Returns
    -------
    None

    """

    header = fits.getheader(im_name)
    shape = (header['NAXIS2'],header['NAXIS1'])

    if step_size is None:
        if 'BMAJ' in header and 'BMIN' in header:
            beam_size = np.sqrt(abs(header['BMAJ']*header['BMIN']))
            if 'CDELT1' in header:
                pix_scale = np.sqrt(abs(header['CDELT1']*header['CDELT2']))
            elif 'CD1_1' in header:
                pix_scale = np.sqrt(abs(header['CD1_1']*header['CD2_2']))
                if 'CD1_2' in header and 'CD2_1' in header:
                    if header['CD1_2'] != 0 or header['CD2_1']!=0:
                        logging.warning("CD1_2 and/or CD2_1 are non-zero and I don't know what to do with them")
                        logging.warning("Ingoring them")
            else:
                logging.warning("Cannot determine pixel scale, assuming 4 pixels per beam")
                pix_scale = beam_size/4.
            # default to 4x the synthesized beam width
            step_size = int(np.ceil(4*beam_size/pix_scale))
        else:
            logging.info("BMAJ and/or BMIN not in fits header.")
            logging.info("Assuming 4 pix/beam, so we have step_size = 16 pixels")
            step_size = 16
        step_size = (step_size, step_size)

    if box_size is None:
        # default to 6x the step size so we have ~ 30beams
        box_size = (step_size[0]*6, step_size[1]*6)

    if compressed:
        if not step_size[0] == step_size[1]:
            step_size = (min(step_size), min(step_size))
            logging.info("Changing grid to be {0} so we can compress the output".format(step_size))

    logging.info("using grid_size {0}, box_size {1}".format(step_size,box_size))
    logging.info("on data shape {0}".format(shape))
    bkg, rms = filter_mc_sharemem(im_name, step_size=step_size, box_size=box_size, cores=cores, shape=shape, nslice=nslice, domask=mask)
    logging.info("done")

    bkg_out = '_'.join([os.path.expanduser(out_base), 'bkg.fits'])
    rms_out = '_'.join([os.path.expanduser(out_base), 'rms.fits'])


    # add a comment to the fits header
    header['HISTORY'] = 'BANE {0}-({1})'.format(__version__, __date__)

    # compress
    if compressed:
        hdu = fits.PrimaryHDU(bkg)
        hdu.header = copy.deepcopy(header)
        hdulist = fits.HDUList([hdu])
        compress(hdulist, step_size[0], bkg_out)
        hdulist[0].header = copy.deepcopy(header)
        hdulist[0].data = rms
        compress(hdulist, step_size[0], rms_out)
        return

    write_fits(bkg, header, bkg_out)
    write_fits(rms, header, rms_out)