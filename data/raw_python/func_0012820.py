def onlylegalchar(name):
    """return only legal chars"""
    legalchar = ascii_letters + digits + ' '
    return ''.join([s for s in name[:] if s in legalchar])