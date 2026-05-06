def makecsvdiffs(thediffs, dtls, n1, n2):
    """return the csv to be displayed"""
    def ishere(val):
        if val == None:
            return "not here"
        else:
            return "is here"
    rows = []
    rows.append(['file1 = %s' % (n1, )])
    rows.append(['file2 = %s' % (n2, )])
    rows.append('')
    rows.append(theheader(n1, n2))
    keys = list(thediffs.keys()) # ensures sorting by Name
    keys.sort()
    # sort the keys in the same order as in the idd
    dtlssorter = DtlsSorter(dtls)
    keys = sorted(keys, key=dtlssorter.getkey)
    for key in keys:
        if len(key) == 2:
            rw2 = [''] + [ishere(i) for i in thediffs[key]]
        else:
            rw2 = list(thediffs[key])
        rw1 = list(key)
        rows.append(rw1 + rw2)
    return rows