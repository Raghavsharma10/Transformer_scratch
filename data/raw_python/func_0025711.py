def sigStrToKwArgsDict(checkFuncSig):
    """ Take a check function signature (string), and parse it to get a dict
        of the keyword args and their values. """
    p1 = checkFuncSig.find('(')
    p2 = checkFuncSig.rfind(')')
    assert p1 > 0 and p2 > 0 and p2 > p1, "Invalid signature: "+checkFuncSig
    argParts = irafutils.csvSplit(checkFuncSig[p1+1:p2], ',', True)
    argParts = [x.strip() for x in argParts]
    retval = {}
    for argPair in argParts:
        argSpl = argPair.split('=', 1)
        if len(argSpl) > 1:
            if argSpl[0] in retval:
                if isinstance(retval[argSpl[0]], (list,tuple)):
                    retval[argSpl[0]]+=(irafutils.stripQuotes(argSpl[1]),) # 3rd
                else: # 2nd in, so convert to tuple
                    retval[argSpl[0]] = (retval[argSpl[0]],
                                         irafutils.stripQuotes(argSpl[1]),)
            else:
                retval[argSpl[0]] = irafutils.stripQuotes(argSpl[1]) # 1st in
        else:
            retval[argSpl[0]] = None #  eg. found "triggers=, max=6, ..."
    return retval