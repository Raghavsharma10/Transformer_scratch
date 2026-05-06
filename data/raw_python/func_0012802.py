def pickledump(theobject, fname):
    """same as pickle.dump(theobject, fhandle).takes filename as parameter"""
    fhandle = open(fname, 'wb')
    pickle.dump(theobject, fhandle)