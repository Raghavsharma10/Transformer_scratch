def missingkeys_standard(commdct, dtls, skiplist=None):
    """put missing keys in commdct for standard objects
    return a list of keys where it is unable to do so
    commdct is not returned, but is updated"""
    if skiplist == None:
        skiplist = []
    # find objects where all the fields are not named
    gkeys = [dtls[i] for i in range(len(dtls)) if commdct[i].count({}) > 2]
    nofirstfields = []
    # operatie on those fields
    for key_txt in gkeys:
        if key_txt in skiplist:
            continue
        # print key_txt
        # for a function, pass comm as a variable
        key_i = dtls.index(key_txt.upper())
        comm = commdct[key_i]



        # get all fields
        fields = getfields(comm)

        # get repeating field names
        repnames = repeatingfieldsnames(fields)

        try:
            first = repnames[0][0] % (1, )
        except IndexError:
            nofirstfields.append(key_txt)
            continue
        # print first

        # get all comments of the first repeating field names
        firstnames = [repname[0] % (1, ) for repname in repnames]
        fcomments = [field for field in fields
                     if bunchhelpers.onlylegalchar(field['field'][0])
                     in firstnames]
        fcomments = [dict(fcomment) for fcomment in fcomments]
        for cmt in fcomments:
            fld = cmt['field'][0]
            fld = bunchhelpers.onlylegalchar(fld)
            fld = bunchhelpers.replaceint(fld)
            cmt['field'] = [fld]

        for i, cmt in enumerate(comm[1:]):
            thefield = cmt['field'][0]
            thefield = bunchhelpers.onlylegalchar(thefield)
            if thefield == first:
                break
        first_i = i + 1

        newfields = []
        for i in range(1, len(comm[first_i:]) // len(repnames) + 1):
            for fcomment in fcomments:
                nfcomment = dict(fcomment)
                fld = nfcomment['field'][0]
                fld = fld % (i, )
                nfcomment['field'] = [fld]
                newfields.append(nfcomment)

        for i, cmt in enumerate(comm):
            if i < first_i:
                continue
            else:
                afield = newfields.pop(0)
                comm[i] = afield
        commdct[key_i] = comm
    return nofirstfields