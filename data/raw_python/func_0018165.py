def update_dq(filename, ext, mask, dqval=16384, verbose=True):
    """Update the given image and DQ extension with the given
    satellite trails mask and flag.

    Parameters
    ----------
    filename : str
        FITS image filename to update.

    ext : int, str, or tuple
        DQ extension, as accepted by ``astropy.io.fits``, to update.

    mask : ndarray
        Boolean mask, with `True` marking the satellite trail(s).
        This can be the result(s) from :func:`make_mask`.

    dqval : int, optional
        DQ value to use for the trail. Default value of 16384 is
        tailored for ACS/WFC.

    verbose : bool, optional
        Print extra information to the terminal.

    """
    with fits.open(filename, mode='update') as pf:
        dqarr = pf[ext].data
        old_mask = (dqval & dqarr) != 0  # Existing flagged trails
        new_mask = mask & ~old_mask  # Only flag previously unflagged trails
        npix_updated = np.count_nonzero(new_mask)

        # Update DQ extension only if necessary
        if npix_updated > 0:
            pf[ext].data[new_mask] += dqval
            pf['PRIMARY'].header.add_history('{0} satdet v{1}({2})'.format(
                time.ctime(), __version__, __vdate__))
            pf['PRIMARY'].header.add_history(
                '  Updated {0} px in EXT {1} with DQ={2}'.format(
                    npix_updated, ext, dqval))

    if verbose:
        fname = '{0}[{1}]'.format(filename, ext)

        print('DQ flag value is {0}'.format(dqval))
        print('Input... flagged NPIX={0}'.format(np.count_nonzero(mask)))
        print('Existing flagged NPIX={0}'.format(np.count_nonzero(old_mask)))
        print('Newly... flagged NPIX={0}'.format(npix_updated))

        if npix_updated > 0:
            print('{0} updated'.format(fname))
        else:
            print('No updates necessary for {0}'.format(fname))