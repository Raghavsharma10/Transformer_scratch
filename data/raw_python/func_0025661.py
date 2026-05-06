def buildRootname(filename, ext=None):
    """
    Build a new rootname for an existing file and given extension.

    Any user supplied extensions to use for searching for file need to be
    provided as a list of extensions.

    Examples
    --------

    ::

        >>> rootname = buildRootname(filename, ext=['_dth.fits'])  # doctest: +SKIP

    """

    if filename in ['' ,' ', None]:
        return None

    fpath, fname = os.path.split(filename)
    if ext is not None and '_' in ext[0]:
        froot = os.path.splitext(fname)[0].split('_')[0]
    else:
        froot = fname

    if fpath in ['', ' ', None]:
        fpath = os.curdir
    # Get complete list of filenames from current directory
    flist = os.listdir(fpath)

    #First, assume given filename is complete and verify
    # it exists...
    rootname = None

    for name in flist:
        if name == froot:
            rootname = froot
            break
        elif name == froot + '.fits':
            rootname = froot + '.fits'
            break

    # If we have an incomplete filename, try building a default
    # name and seeing if it exists...
    #
    # Set up default list of suffix/extensions to add to rootname
    _extlist = []
    for extn in EXTLIST:
        _extlist.append(extn)

    if rootname is None:
        # Add any user-specified extension to list of extensions...
        if ext is not None:
            for i in ext:
                _extlist.insert(0,i)
        # loop over all extensions looking for a filename that matches...
        for extn in _extlist:
            # Start by looking for filename with exactly
            # the same case a provided in ASN table...
            rname = froot + extn
            for name in flist:
                if rname == name:
                    rootname = name
                    break
            if rootname is None:
                # Try looking for all lower-case filename
                # instead of a mixed-case filename as required
                # by the pipeline.
                rname = froot.lower() + extn
                for name in flist:
                    if rname == name:
                        rootname = name
                        break

            if rootname is not None:
                break

    # If we still haven't found the file, see if we have the
    # info to build one...
    if rootname is None and ext is not None:
        # Check to see if we have a full filename to start with...
        _indx = froot.find('.')
        if _indx > 0:
            rootname = froot[:_indx] + ext[0]
        else:
            rootname = froot + ext[0]

    if fpath not in ['.', '', ' ', None]:
        rootname = os.path.join(fpath, rootname)
    # It will be up to the calling routine to verify
    # that a valid rootname, rather than 'None', was returned.
    return rootname