def idd2grouplist(fhandle):
    """wrapper for iddtxt2grouplist"""
    try:
        txt = fhandle.read()
        return iddtxt2grouplist(txt)
    except AttributeError as e:
        txt = open(fhandle, 'r').read()
        return iddtxt2grouplist(txt)