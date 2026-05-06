def slistFloat(slist):
    """ Converts signed list to float. """
    values = [v / 60**(i) for (i,v) in enumerate(slist[1:])]
    value = sum(values)
    return -value if slist[0] == '-' else value