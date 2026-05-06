def idd2group(fhandle):
    """wrapper for iddtxt2groups"""
    try:
        txt = fhandle.read()
        return iddtxt2groups(txt)
    except AttributeError as e:
        txt = open(fhandle, 'r').read()
        return iddtxt2groups(txt)