def getmakeidfobject(idf, key, name):
    """get idfobject or make it if it does not exist"""
    idfobject = idf.getobject(key, name)
    if not idfobject:
        return idf.newidfobject(key, Name=name)
    else:
        return idfobject