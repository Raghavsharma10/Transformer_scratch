def iddofobject(data, commdct, key):
    """from commdct, return the idd of the object key"""
    dtls = data.dtls
    i = dtls.index(key)
    return commdct[i]