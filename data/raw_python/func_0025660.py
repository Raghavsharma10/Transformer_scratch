def buildNewRootname(filename, extn=None, extlist=None):
    """
    Build rootname for a new file.

    Use 'extn' for new filename if given, does NOT append a suffix/extension at
    all.

    Does NOT check to see if it exists already.  Will ALWAYS return a new
    filename.
    """

    # Search known suffixes to replace ('_crj.fits',...)
    _extlist = copy.deepcopy(EXTLIST)
    # Also, add a default where '_dth.fits' replaces
    # whatever extension was there ('.fits','.c1h',...)
    #_extlist.append('.')
    # Also append any user-specified extensions...
    if extlist:
        _extlist += extlist

    if isinstance(filename, fits.HDUList):
        try:
            filename = filename.filename()
        except:
            raise ValueError("Can't determine the filename of an waivered HDUList object.")
    for suffix in _extlist:
        _indx = filename.find(suffix)
        if _indx > 0: break

    if _indx < 0:
         # default to entire rootname
        _indx = len(filename)

    if extn is None:
        extn = ''

    return filename[:_indx] + extn