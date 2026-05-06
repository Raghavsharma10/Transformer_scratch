def write_str2file(pathname, astr):
    """writes a string to file"""
    fname = pathname
    fhandle = open(fname, 'wb')
    fhandle.write(astr)
    fhandle.close()