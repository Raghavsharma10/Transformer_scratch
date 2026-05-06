def iddtxt2grouplist(txt):
    """return a list of group names
    the list in the same order as the idf objects in idd file
    """
    def makenone(astr):
        if astr == 'None':
            return None
        else:
            return astr

    txt = nocomment(txt, '!')
    txt = txt.replace("\\group", "!-group") # retains group in next line
    txt = nocomment(txt, '\\') # remove all other idd info
    lines = txt.splitlines()
    lines = [line.strip() for line in lines] # cleanup
    lines = [line for line in lines if line != ''] # cleanup
    txt = '\n'.join(lines)
    gsplits = txt.split('!') # split into groups, since we have !-group
    gsplits = [gsplit.splitlines() for gsplit in gsplits] # split group

    gsplits[0].insert(0, u'-group None')
        # Put None for the first group that does nothave a group name
    
    glist = []
    for gsplit in gsplits:
        glist.append((gsplit[0], gsplit[1:]))
        # makes dict {groupname:[k1, k2], groupname2:[k3, k4]}

    glist = [(k, '\n'.join(v)) for k, v in glist]# joins lines back
    glist = [(k, v.split(';')) for k, v in glist] # splits into idfobjects
    glist = [(k, [i.strip() for i in v]) for k, v in glist] # cleanup
    glist = [(k, [i.splitlines() for i in v]) for k, v in glist]
        # splits idfobjects into lines
    glist = [(k, [i for i in v if len(i) > 0]) for k, v in glist]
        # cleanup - removes blank lines
    glist = [(k, [i[0] for i in v]) for k, v in glist] # use first line
    fglist = []
    for gnamelist in glist:
        gname = gnamelist[0]
        thelist = gnamelist[-1]
        for item in thelist:
            fglist.append((gname, item))
    glist = [(gname[len("-group "):], obj) for gname, obj in fglist] # remove "-group "
    glist = [(makenone(gname), obj) for gname, obj in glist] # make str None into None
    glist = [(gname, obj.split(',')[0]) for gname, obj in glist] # remove comma
    return glist