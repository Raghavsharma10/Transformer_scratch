def separateKeywords(kwArgsDict):
    """ Look through the keywords passed and separate the special ones we
        have added from the legal/standard ones.  Return both sets as two
        dicts (in a tuple), as (standardKws, ourKws) """
    standardKws = {}
    ourKws = {}
    for k in kwArgsDict:
        if k in STANDARD_KEYS:
            standardKws[k]=kwArgsDict[k]
        else:
            ourKws[k]=kwArgsDict[k]
    return (standardKws, ourKws)