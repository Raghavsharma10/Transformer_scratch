def _roundSlist(slist):
    """ Rounds a signed list over the last element and removes it. """
    slist[-1] = 60 if slist[-1] >= 30 else 0
    for i in range(len(slist)-1, 1, -1):
        if slist[i] == 60:
            slist[i] = 0
            slist[i-1] += 1
    return slist[:-1]