def listVars(prefix="", equals="\t= ", **kw):
    """List IRAF variables."""

    keylist = getVarList()
    if len(keylist) == 0:
        print('No IRAF variables defined')
    else:
        keylist.sort()
        for word in keylist:
            print("%s%s%s%s" % (prefix, word, equals, envget(word)))