def isHiddenName(astr):
    """ Return True if this string name denotes a hidden par or section """
    if astr is not None and len(astr) > 2 and astr.startswith('_') and \
       astr.endswith('_'):
        return True
    else:
        return False