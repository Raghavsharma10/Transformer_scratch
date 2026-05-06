def findFile(input):
    """Search a directory for full filename with optional path."""

    # If no input name is provided, default to returning 'no'(FALSE)
    if not input:
        return no

    # We use 'osfn' here to insure that any IRAF variables are
    # expanded out before splitting out the path...
    _fdir, _fname = os.path.split(osfn(input))

    if _fdir == '':
        _fdir = os.curdir

    try:
        flist = os.listdir(_fdir)
    except OSError:
        # handle when requested file in on a disconnect network store
        return no

    _root, _extn = parseFilename(_fname)

    found = no
    for name in flist:
        if name == _root:
            # Check to see if given extension, if any, exists
            if _extn is None:
                found = yes
                continue
            else:
                _split = _extn.split(',')
                _extnum = None
                _extver = None
                if  _split[0].isdigit():
                    _extname = None
                    _extnum = int(_split[0])
                else:
                    _extname = _split[0]
                    if len(_split) > 1:
                        _extver = int(_split[1])
                    else:
                        _extver = 1
                f = openImage(_root)
                f.close()
                if _extnum is not None:
                    if _extnum < len(f):
                        found = yes
                        del f
                        continue
                    else:
                        del f
                else:
                    _fext = findExtname(f, _extname, extver=_extver)
                    if _fext is not None:
                        found = yes
                        del f
                        continue
    return found