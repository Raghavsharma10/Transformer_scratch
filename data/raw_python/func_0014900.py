def strToTempfile(s, suffix=None, prefix=None, dir=None, binary=False):
    """Create a new tempfile, write ``s`` to it and return the filename.
    `suffix`, `prefix` and `dir` are like in `tempfile.mkstemp`.
    """
    fd, filename = tempfile.mkstemp(**dict((k,v) for (k,v) in
                                           [('suffix',suffix),('prefix',prefix),('dir', dir)]
                                           if v is not None))
    spitOut(s, fd, binary)
    return filename