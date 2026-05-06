def getHeader(filename, handle=None):
    """
    Return a copy of the PRIMARY header, along with any group/extension header
    for this filename specification.
    """

    _fname, _extn = parseFilename(filename)
    # Allow the user to provide an already opened PyFITS object
    # to derive the header from...
    #
    if not handle:
        # Open image whether it is FITS or GEIS
        _fimg = openImage(_fname, mode='readonly')
    else:
        # Use what the user provides, after insuring
        # that it is a proper PyFITS object.
        if isinstance(handle, fits.HDUList):
            _fimg = handle
        else:
            raise ValueError('Handle must be a %r object!' % fits.HDUList)

    _hdr = _fimg['PRIMARY'].header.copy()

    # if the data is not in the primary array delete NAXIS
    # so that the correct value is read from the extension header
    if _hdr['NAXIS'] == 0:
        del _hdr['NAXIS']

    if not (_extn is None or (_extn.isdigit() and int(_extn) == 0)):
        # Append correct extension/chip/group header to PRIMARY...
        #for _card in getExtn(_fimg,_extn).header.ascard:
            #_hdr.ascard.append(_card)
        for _card in getExtn(_fimg, _extn).header.cards:
            _hdr.append(_card)
    if not handle:
        # Close file handle now...
        _fimg.close()
        del _fimg

    return _hdr