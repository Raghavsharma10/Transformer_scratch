def printdict(adict):
    """printdict"""
    dlist = list(adict.keys())
    dlist.sort()
    for i in range(0, len(dlist)):
        print(dlist[i], adict[dlist[i]])