def _readfname(fname):
    """copied from extractidddata below. 
    It deals with all the types of fnames"""
    try:
        if isinstance(fname, (file, StringIO)):
            astr = fname.read()
        else:
            astr = open(fname, 'rb').read()
    except NameError:
        if isinstance(fname, (FileIO, StringIO)):
            astr = fname.read()
        else:
            astr = mylib2.readfile(fname)
    return astr