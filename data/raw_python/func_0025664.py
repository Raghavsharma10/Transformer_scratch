def updateKeyword(filename, key, value,show=yes):
    """Add/update keyword to header with given value."""

    _fname, _extn = parseFilename(filename)

    # Open image whether it is FITS or GEIS
    _fimg = openImage(_fname, mode='update')

    # Address the correct header
    _hdr = getExtn(_fimg, _extn).header

    # Assign a new value or add new keyword here.
    try:
        _hdr[key] = value
    except KeyError:
        if show:
            print('Adding new keyword ', key, '=', value)
        _hdr[key] = value

    # Close image
    _fimg.close()
    del _fimg