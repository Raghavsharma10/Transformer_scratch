def cleaniddfield(acomm):
    """make all the keys lower case"""
    for key in list(acomm.keys()):
        val = acomm[key]
        acomm[key.lower()] = val
    for key in list(acomm.keys()):
        val = acomm[key]
        if key != key.lower():
            acomm.pop(key)
    return acomm