def getobjects(bunchdt, data, commdct, key, places=7, **kwargs):
    """get all the objects of key that matches the fields in **kwargs"""
    idfobjects = bunchdt[key]
    allobjs = []
    for obj in idfobjects:
        if __objecthasfields(
                bunchdt, data, commdct,
                obj, places=places, **kwargs):
            allobjs.append(obj)
    return allobjs