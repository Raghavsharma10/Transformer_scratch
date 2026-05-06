def geis2mef(sciname, convert_dq=True):
    """
    Converts a GEIS science file and its corresponding
    data quality file (if present) to MEF format
    Writes out both files to disk.
    Returns the new name of the science image.
    """
    clobber = True
    mode = 'update'
    memmap = True
    # Input was specified as a GEIS image, but no FITS copy
    # exists.  Read it in with 'readgeis' and make a copy
    # then open the FITS copy...
    try:
        # Open as a GEIS image for reading only
        fimg = readgeis.readgeis(sciname)
    except Exception:
        raise IOError("Could not open GEIS input: %s" % sciname)

    #check for the existence of a data quality file
    _dqname = fileutil.buildNewRootname(sciname, extn='.c1h')
    dqexists = os.path.exists(_dqname)
    if dqexists:
        try:
            dqfile = readgeis.readgeis(_dqname)
            dqfitsname = fileutil.buildFITSName(_dqname)
        except Exception:
            print("Could not read data quality file %s" % _dqname)

    # Check to see if user wanted to update GEIS header.
    # or write out a multi-extension FITS file and return a handle to it
    # User wants to make a FITS copy and update it
    # using the filename they have provided
    fitsname = fileutil.buildFITSName(sciname)

    # Write out GEIS image as multi-extension FITS.
    fexists = os.path.exists(fitsname)
    if (fexists and clobber) or not fexists:
            print('Writing out GEIS as MEF to ', fitsname)
            if ASTROPY_VER_GE13:
                fimg.writeto(fitsname, overwrite=clobber)
            else:
                fimg.writeto(fitsname, clobber=clobber)
            if dqexists:
                print('Writing out GEIS as MEF to ', dqfitsname)
                if ASTROPY_VER_GE13:
                    dqfile.writeto(dqfitsname, overwrite=clobber)
                else:
                    dqfile.writeto(dqfitsname, clobber=clobber)
    # Now close input GEIS image, and open writable
    # handle to output FITS image instead...
    fimg.close()
    del fimg
    fimg = fits.open(fitsname, mode=mode, memmap=memmap)

    return fimg