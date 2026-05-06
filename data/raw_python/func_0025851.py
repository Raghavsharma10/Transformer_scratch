def parseFilename(filename):
    """
        Parse out filename from any specified extensions.
        Returns rootname and string version of extension name.

        Modified from 'pydrizzle.fileutil' to allow this
        module to be independent of PyDrizzle/MultiDrizzle.

    """
    # Parse out any extension specified in filename
    _indx = filename.find('[')
    if _indx > 0:
        # Read extension name provided
        _fname = filename[:_indx]
        extn = filename[_indx+1:-1]

        # An extension was provided, so parse it out...
        if repr(extn).find(',') > 1:
            _extns = extn.split(',')
            # Two values given for extension:
            #    for example, 'sci,1' or 'dq,1'
            _extn = [_extns[0],int(_extns[1])]
        elif repr(extn).find('/') > 1:
            # We are working with GEIS group syntax
            _indx = str(extn[:extn.find('/')])
            _extn = [int(_indx)]
        elif isinstance(extn, str):
            # Only one extension value specified...
            if extn.isdigit():
                # We only have an extension number specified as a string...
                _nextn = int(extn)
            else:
                # We only have EXTNAME specified...
                _nextn = extn
            _extn = [_nextn]
        else:
            # Only integer extension number given, or default of 0 is used.
            _extn = [int(extn)]

    else:
        _fname = filename
        _extn = None
    return _fname,_extn