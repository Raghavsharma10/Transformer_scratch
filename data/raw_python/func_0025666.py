def openImage(filename, mode='readonly', memmap=False, writefits=True,
              clobber=True, fitsname=None):
    """
    Opens file and returns PyFITS object.  Works on both FITS and GEIS
    formatted images.

    Notes
    -----
    If a GEIS or waivered FITS image is used as input, it will convert it to a
    MEF object and only if ``writefits = True`` will write it out to a file. If
    ``fitsname = None``, the name used to write out the new MEF file will be
    created using `buildFITSName`.

    Parameters
    ----------
    filename: str
        name of input file
    mode: str
        mode for opening file based on PyFITS `mode` parameter values
    memmap: bool
        switch for using memory mapping, `False` for no, `True` for yes
    writefits: bool
        if `True`, will write out GEIS as multi-extension FITS
        and return handle to that opened GEIS-derived MEF file
    clobber: bool
        overwrite previously written out GEIS-derived MEF file
    fitsname: str
        name to use for GEIS-derived MEF file,
        if None and writefits==`True`, will use 'buildFITSName()' to generate one
    """
    if not isinstance(filename, fits.HDUList):
        # Insure that the filename is always fully expanded
        # This will not affect filenames without paths or
        # filenames specified with extensions.
        filename = osfn(filename)

        # Extract the rootname and extension specification
        # from input image name
        _fname, _iextn = parseFilename(filename)
    else:
        _fname = filename

    # Check whether we have a FITS file and if so what type
    isfits, fitstype = isFits(_fname)

    if isfits:
        if fitstype != 'waiver':
            # Open the FITS file
            fimg = fits.open(_fname, mode=mode, memmap=memmap)
            return fimg
        else:
            fimg = convertwaiveredfits.convertwaiveredfits(_fname)

            #check for the existence of a data quality file
            _dqname = buildNewRootname(_fname, extn='_c1f.fits')
            dqexists = os.path.exists(_dqname)
            if dqexists:
                try:
                    dqfile = convertwaiveredfits.convertwaiveredfits(_dqname)
                    dqfitsname = buildNewRootname(_dqname, extn='_c1h.fits')
                except:
                    print("Could not read data quality file %s" % _dqname)
            if writefits:
                # User wants to make a FITS copy and update it
                # using the filename they have provided
                if fitsname is None:
                    rname = buildNewRootname(_fname)
                    fitsname = buildNewRootname(rname, extn='_c0h.fits')

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
                # Image re-written as MEF, now it needs its WCS updated
                #updatewcs.updatewcs(fitsname)

                fimg = fits.open(fitsname, mode=mode, memmap=memmap)

        # Return handle for use by user
        return fimg
    else:
        # Input was specified as a GEIS image, but no FITS copy
        # exists.  Read it in with 'readgeis' and make a copy
        # then open the FITS copy...
        try:
            # Open as a GEIS image for reading only
            fimg = readgeis.readgeis(_fname)
        except:
            raise IOError("Could not open GEIS input: %s" % _fname)

        #check for the existence of a data quality file
        _dqname = buildNewRootname(_fname, extn='.c1h')
        dqexists = os.path.exists(_dqname)
        if dqexists:
            try:
                dqfile = readgeis.readgeis(_dqname)
                dqfitsname = buildFITSName(_dqname)
            except:
                print("Could not read data quality file %s" % _dqname)

        # Check to see if user wanted to update GEIS header.
        # or write out a multi-extension FITS file and return a handle to it
        if writefits:
                # User wants to make a FITS copy and update it
                # using the filename they have provided
            if fitsname is None:
                fitsname = buildFITSName(_fname)

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
            # Image re-written as MEF, now it needs its WCS updated
            #updatewcs.updatewcs(fitsname)

            fimg = fits.open(fitsname, mode=mode, memmap=memmap)

        # Return handle for use by user
        return fimg