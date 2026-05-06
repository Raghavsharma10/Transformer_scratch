def waiver2mef(sciname, newname=None, convert_dq=True, writefits=True):
    """
    Converts a GEIS science file and its corresponding
    data quality file (if present) to MEF format
    Writes out both files to disk.
    Returns the new name of the science image.
    """

    if isinstance(sciname, fits.HDUList):
        filename = sciname.filename()
    else:
        filename = sciname

    try:
        clobber = True
        fimg = convertwaiveredfits.convertwaiveredfits(filename)

        #check for the existence of a data quality file
        _dqname = fileutil.buildNewRootname(filename, extn='_c1f.fits')
        dqexists = os.path.exists(_dqname)
        if convert_dq and dqexists:
            try:
                dqfile = convertwaiveredfits.convertwaiveredfits(_dqname)
                dqfitsname = fileutil.buildNewRootname(_dqname, extn='_c1h.fits')
            except Exception:
                print("Could not read data quality file %s" % _dqname)
        if writefits:
            # User wants to make a FITS copy and update it
            # using the filename they have provided
            rname = fileutil.buildNewRootname(filename)
            fitsname = fileutil.buildNewRootname(rname, extn='_c0h.fits')

            # Write out GEIS image as multi-extension FITS.
            fexists = os.path.exists(fitsname)
            if (fexists and clobber) or not fexists:
                print('Writing out WAIVERED as MEF to ', fitsname)
                if ASTROPY_VER_GE13:
                    fimg.writeto(fitsname, overwrite=clobber)
                else:
                    fimg.writeto(fitsname, clobber=clobber)
                if dqexists:
                    print('Writing out WAIVERED as MEF to ', dqfitsname)
                    if ASTROPY_VER_GE13:
                        dqfile.writeto(dqfitsname, overwrite=clobber)
                    else:
                        dqfile.writeto(dqfitsname, clobber=clobber)
        # Now close input GEIS image, and open writable
        # handle to output FITS image instead...
        fimg.close()
        del fimg

        fimg = fits.open(fitsname, mode='update', memmap=False)

        return fimg
    except IOError:
        print('Warning: File %s could not be found' % sciname)
        return None