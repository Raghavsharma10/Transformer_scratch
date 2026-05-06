def makedoetree(ddict, bdict):
    """makedoetree"""
    dlist = list(ddict.keys())
    blist = list(bdict.keys())
    dlist.sort()
    blist.sort()
    #make space dict
    doesnot = 'DOES NOT'
    lst = []
    for num in range(0, len(blist)):
        if bdict[blist[num]] == doesnot:#belong
            lst = lst + [blist[num]]

    doedict = {}
    for num in range(0, len(lst)):
        #print lst[num]
        doedict[lst[num]] = {}
    lv1list = list(doedict.keys())
    lv1list.sort()

    #make wall dict
    #for each space
    for i in range(0, len(lv1list)):
        walllist = []
        adict = doedict[lv1list[i]]
        #loop thru the entire blist dictonary and list the ones that belong into walllist
        for num in range(0, len(blist)):
            if bdict[blist[num]] == lv1list[i]:
                walllist = walllist + [blist[num]]
        #put walllist into dict
        for j in range(0, len(walllist)):
            adict[walllist[j]] = {}

    #make window dict
    #for each space
    for i in range(0, len(lv1list)):
        adict1 = doedict[lv1list[i]]
        #for each wall
        walllist = list(adict1.keys())
        walllist.sort()
        for j in range(0, len(walllist)):
            windlist = []
            adict2 = adict1[walllist[j]]
           #loop thru the entire blist dictonary and list the ones that belong into windlist
            for num in range(0, len(blist)):
                if bdict[blist[num]] == walllist[j]:
                    windlist = windlist + [blist[num]]
            #put walllist into dict
            for k in range(0, len(windlist)):
                adict2[windlist[k]] = {}
    return doedict