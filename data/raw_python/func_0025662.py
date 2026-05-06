def getKeyword(filename, keyword, default=None, handle=None):
    """
    General, write-safe method for returning a keyword value from the header of
    a IRAF recognized image.

    Returns the value as a string.
    """
    # Insure that there is at least 1 extension specified...
    if filename.find('[') < 0:
        filename += '[0]'

    _fname, _extn = parseFilename(filename)

    if not handle:
        # Open image whether it is FITS or GEIS
        _fimg = openImage(_fname)
    else:
        # Use what the user provides, after insuring
        # that it is a proper PyFITS object.
        if isinstance(handle, fits.HDUList):
            _fimg = handle
        else:
            raise ValueError('Handle must be %r object!' % fits.HDUList)

    # Address the correct header
    _hdr = getExtn(_fimg, _extn).header

    try:
        value =  _hdr[keyword]
    except KeyError:
        _nextn = findKeywordExtn(_fimg, keyword)
        try:
            value = _fimg[_nextn].header[keyword]
        except KeyError:
            value = ''

    if not handle:
        _fimg.close()
        del _fimg

    if value == '':
        if default is None:
            value = None
        else:
            value = default

    # NOTE:  Need to clean up the keyword.. Occasionally the keyword value
    # goes right up to the "/" FITS delimiter, and iraf.keypar is incapable
    # of realizing this, so it incorporates "/" along with the keyword value.
    # For example, after running "pydrizzle" on the image "j8e601bkq_flt.fits",
    # the CD keywords look like this:
    #
    #   CD1_1   = 9.221627430999639E-06/ partial of first axis coordinate w.r.t. x
    #   CD1_2   = -1.0346992614799E-05 / partial of first axis coordinate w.r.t. y
    #
    # so for CD1_1, iraf.keypar returns:
    #       "9.221627430999639E-06/"
    #
    # So, the following piece of code CHECKS for this and FIXES the string,
    # very simply by removing the last character if it is a "/".
    # This fix courtesy of Anton Koekemoer, 2002.
    elif isinstance(value, string_types):
        if value[-1:] == '/':
            value = value[:-1]

    return value