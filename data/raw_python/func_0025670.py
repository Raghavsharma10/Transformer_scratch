def getExtn(fimg, extn=None):
    """
    Returns the PyFITS extension corresponding to extension specified in
    filename.

    Defaults to returning the first extension with data or the primary
    extension, if none have data.  If a non-existent extension has been
    specified, it raises a `KeyError` exception.
    """

    # If no extension is provided, search for first extension
    # in FITS file with data associated with it.
    if extn is None:
        # Set up default to point to PRIMARY extension.
        _extn = fimg[0]
        # then look for first extension with data.
        for _e in fimg:
            if _e.data is not None:
                _extn = _e
                break
    else:
        # An extension was provided, so parse it out...
        if repr(extn).find(',') > 1:
            if isinstance(extn, tuple):
                # We have a tuple possibly created by parseExtn(), so
                # turn it into a list for easier manipulation.
                _extns = list(extn)
                if '' in _extns:
                    _extns.remove('')
            else:
                _extns = extn.split(',')
            # Two values given for extension:
            #    for example, 'sci,1' or 'dq,1'
            try:
                _extn = fimg[_extns[0], int(_extns[1])]
            except KeyError:
                _extn = None
                for e in fimg:
                    hdr = e.header
                    if ('extname' in hdr and
                            hdr['extname'].lower() == _extns[0].lower() and
                            hdr['extver'] == int(_extns[1])):
                        _extn = e
                        break
        elif repr(extn).find('/') > 1:
            # We are working with GEIS group syntax
            _indx = str(extn[:extn.find('/')])
            _extn = fimg[int(_indx)]
        elif isinstance(extn, string_types):
            if extn.strip() == '':
                _extn = None  # force error since invalid name was provided
            # Only one extension value specified...
            elif extn.isdigit():
                # We only have an extension number specified as a string...
                _nextn = int(extn)
            else:
                # We only have EXTNAME specified...
                _nextn = None
                if extn.lower() == 'primary':
                    _nextn = 0
                else:
                    i = 0
                    for hdu in fimg:
                        isimg = 'extname' in hdu.header
                        hdr = hdu.header
                        if isimg and extn.lower() == hdr['extname'].lower():
                            _nextn = i
                            break
                        i += 1

            if _nextn < len(fimg):
                _extn = fimg[_nextn]
            else:
                _extn = None

        else:
            # Only integer extension number given, or default of 0 is used.
            if int(extn) < len(fimg):
                _extn = fimg[int(extn)]
            else:
                _extn = None

    if _extn is None:
        raise KeyError('Extension %s not found' % extn)

    return _extn