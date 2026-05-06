def _open_fits_files(filenames):
    """
    Given a {correlation: filename} mapping for filenames
    returns a {correlation: file handle} mapping
    """
    kw = { 'mode' : 'update', 'memmap' : False }

    def _fh(fn):
        """ Returns a filehandle or None if file does not exist """
        return fits.open(fn, **kw) if os.path.exists(fn) else None

    return collections.OrderedDict(
            (corr, tuple(_fh(fn) for fn in files))
        for corr, files in filenames.iteritems() )