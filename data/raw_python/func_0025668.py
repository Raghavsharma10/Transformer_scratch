def parseExtn(extn=None):
    """
    Parse a string representing a qualified fits extension name as in the
    output of `parseFilename` and return a tuple ``(str(extname),
    int(extver))``, which can be passed to `astropy.io.fits` functions using
    the 'ext' kw.

    Default return is the first extension in a fits file.

    Examples
    --------

    ::

        >>> parseExtn('sci, 2')
        ('sci', 2)
        >>> parseExtn('2')
        ('', 2)
        >>> parseExtn('sci')
        ('sci', 1)

    """

    if not extn:
        return ('', 0)

    try:
        lext = extn.split(',')
    except:
        return ('', 1)

    if len(lext) == 1 and lext[0].isdigit():
        return ("", int(lext[0]))
    elif len(lext) == 2:
        return (lext[0], int(lext[1]))
    else:
        return (lext[0], 1)