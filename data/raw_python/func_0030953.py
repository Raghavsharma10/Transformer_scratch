def gzip_open_text(path, encoding=None):
    """Opens a plain-text file that may be gzip'ed.

    Parameters
    ----------
    path : str
        The file.
    encoding : str, optional
        The encoding to use.

    Returns
    -------
    file-like
        A file-like object.

    Notes
    -----
    Generally, reading gzip'ed files with gzip.open is very slow, and it is
    preferable to pipe the file into the python script using ``gunzip -c``.
    The script then reads the file from stdin.
    """
    if encoding is None:
        encoding = sys.getdefaultencoding()

    assert os.path.isfile(path)

    is_compressed = False
    try:
        gzip.open(path, mode='rb').read(1)
    except IOError:
        pass
    else:
        is_compressed = True

    if is_compressed:
        if six.PY2:
            import codecs
            zf = gzip.open(path, 'rb')
            reader = codecs.getreader(encoding)
            fh = reader(zf)

        else:
            fh = gzip.open(path, mode='rt', encoding=encoding)

    else:
        # the following works in Python 2.7, thanks to future
        fh = open(path, mode='r', encoding=encoding)

    return fh