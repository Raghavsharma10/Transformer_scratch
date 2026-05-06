def addKwdArgsToSig(sigStr, kwArgsDict):
    """ Alter the passed function signature string to add the given kewords """
    retval = sigStr
    if len(kwArgsDict) > 0:
        retval = retval.strip(' ,)') # open up the r.h.s. for more args
        for k in kwArgsDict:
            if retval[-1] != '(': retval += ", "
            retval += str(k)+"="+str(kwArgsDict[k])
        retval += ')'
    retval = retval
    return retval