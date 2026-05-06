def read(fname):
    """
    utility function to read and return file contents
    """
    fpath = os.path.join(os.path.dirname(__file__), fname)
    with codecs.open(fpath, 'r', 'utf8') as fhandle:
        return fhandle.read().strip()