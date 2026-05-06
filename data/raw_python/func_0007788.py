def slistStr(slist):
    """ Converts signed list to angle string. """
    slist = _fixSlist(slist)
    string = ':'.join(['%02d' % x for x in slist[1:]])
    return slist[0] + string