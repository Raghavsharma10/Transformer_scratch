def objectcount(data, key):
    """return the count of objects of key"""
    objkey = key.upper()
    return len(data.dt[objkey])