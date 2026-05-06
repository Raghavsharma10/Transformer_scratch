def tree2doe(str1):
    """tree2doe"""
    retstuff = makedoedict(str1)
    ddict = makedoetree(retstuff[0], retstuff[1])
    ddict = retstuff[0]
    retstuff[1] = {}# don't need it anymore

    str1 = ''#just re-using it
    l1list = list(ddict.keys())
    l1list.sort()
    for i in range(0, len(l1list)):
        str1 = str1 + ddict[l1list[i]]
        l2list = list(ddict[l1list[i]].keys())
        l2list.sort()
        for j in range(0, len(l2list)):
            str1 = str1 + ddict[l2list[j]]
            l3list = list(ddict[l1list[i]][l2list[j]].keys())
            l3list.sort()
            for k in range(0, len(l3list)):
                str1 = str1 + ddict[l3list[k]]
    return str1