def smart_open_read(path=None, mode='rb', encoding=None, try_gzip=False):
    """Open a file for reading or return ``stdin``.

    Adapted from StackOverflow user "Wolph"
    (http://stackoverflow.com/a/17603000).
    """
    assert mode in ('r', 'rb')
    assert path is None or isinstance(path, (str, _oldstr))
    assert isinstance(mode, (str, _oldstr))
    assert encoding is None or isinstance(encoding, (str, _oldstr))
    assert isinstance(try_gzip, bool)

    fh = None
    binfh = None
    gzfh = None
    if path is None:
        # open stdin
        fh = io.open(sys.stdin.fileno(), mode=mode, encoding=encoding)

    else:
        # open an actual file

        if try_gzip:
            # gzip.open defaults to mode 'rb'
            gzfh = try_open_gzip(path)

        if gzfh is not None:
            logger.debug('Opening gzip''ed file.')
            # wrap gzip stream
            binfh = io.BufferedReader(gzfh)
            if 'b' not in mode:
                # add a text wrapper on top
                logger.debug('Adding text wrapper.')
                fh = io.TextIOWrapper(binfh, encoding=encoding)

        else:
            fh = io.open(path, mode=mode, encoding=encoding)

    yield_fh = fh
    if fh is None:
        yield_fh = binfh

    try:
        yield yield_fh

    finally:
        # close all open files
        if fh is not None:
            # make sure we don't close stdin
            if fh.fileno() != sys.stdin.fileno():
                fh.close()

        if binfh is not None:
            binfh.close()

        if gzfh is not None:
            gzfh.close()