def splitStis(stisfile, sci_count):
    """
    Split a STIS association file into multiple imset MEF files.

    Split the corresponding spt file if present into single spt files.
    If an spt file can't be split or is missing a Warning is printed.

    Returns
    -------
    names: list
        a list with the names of the new flt files.

    """
    newfiles = []

    toclose = False
    if isinstance(stisfile, str):
        f = fits.open(stisfile)
        toclose = True
    else:
        f = stisfile
    hdu0 = f[0].copy()
    stisfilename = stisfile.filename()

    for count in range(1,sci_count+1):
        fitsobj = fits.HDUList()
        fitsobj.append(hdu0)
        hdu = f[('sci',count)].copy()
        fitsobj.append(hdu)
        rootname = hdu.header['EXPNAME']
        newfilename = fileutil.buildNewRootname(rootname, extn='_flt.fits')
        try:
            # Verify error array exists
            if f[('err', count)].data is None:
                raise ValueError
            # Verify dq array exists
            if f[('dq', count)].data is None:
                raise ValueError
            # Copy the err extension
            hdu = f[('err',count)].copy()
            fitsobj.append(hdu)
            # Copy the dq extension
            hdu = f[('dq',count)].copy()
            fitsobj.append(hdu)
            fitsobj[1].header['EXTVER'] = 1
            fitsobj[2].header['EXTVER'] = 1
            fitsobj[3].header['EXTVER'] = 1

        except ValueError:
            print('\nWarning:')
            print('Extension version %d of the input file %s does not' %(count, stisfile))
            print('contain all required image extensions. Each must contain')
            print('populates SCI, ERR and DQ arrays.')

            continue


        # Determine if the file you wish to create already exists on the disk.
        # If the file does exist, replace it.
        if (os.path.exists(newfilename)):
            os.remove(newfilename)
            print("       Replacing "+newfilename+"...")

            # Write out the new file
        fitsobj.writeto(newfilename)
        # Insure returned HDUList is associated with a file
        fitsobj.close()
        fitsobj = fits.open(newfilename, mode='update')
        newfiles.append(fitsobj) # Return HDUList, not filename

    f.close()

    sptfilename = fileutil.buildNewRootname(stisfilename, extn='_spt.fits')
    try:
        sptfile = fits.open(sptfilename)
    except IOError:
        print('SPT file not found %s \n' % sptfilename)
        return newfiles

    if sptfile:
        hdu0 = sptfile[0].copy()
        try:
            for count in range(1,sci_count+1):
                fitsobj = fits.HDUList()
                fitsobj.append(hdu0)
                hdu = sptfile[count].copy()
                fitsobj.append(hdu)
                rootname = hdu.header['EXPNAME']
                newfilename = fileutil.buildNewRootname(rootname, extn='_spt.fits')
                fitsobj[1].header['EXTVER'] = 1
                if (os.path.exists(newfilename)):
                    os.remove(newfilename)
                    print("       Replacing "+newfilename+"...")

                # Write out the new file
                fitsobj.writeto(newfilename)
        except:
            print("Warning: Unable to split spt file %s " % sptfilename)
        if toclose:
            sptfile.close()

    return newfiles