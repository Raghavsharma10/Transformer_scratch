def smart_open_write(path=None, mode='wb', encoding=None):
    """Open a file for writing or return ``stdout``.

    Adapted from StackOverflow user "Wolph"
    (http://stackoverflow.com/a/17603000).
    """
    if path is not None:
        # open a file
        fh = io.open(path, mode=mode, encoding=encoding)
    else:
        # open stdout
        fh = io.open(sys.stdout.fileno(), mode=mode, encoding=encoding)
        #fh = sys.stdout

    try:
        yield fh

    finally:
        # make sure we don't close stdout
        if fh.fileno() != sys.stdout.fileno():
            fh.close()