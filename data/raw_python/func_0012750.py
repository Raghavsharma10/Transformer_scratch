def obj2bunch(data, commdct, obj):
    """make a new bunch object using the data object"""
    dtls = data.dtls
    key = obj[0].upper()
    key_i = dtls.index(key)
    abunch = makeabunch(commdct, obj, key_i)
    return abunch