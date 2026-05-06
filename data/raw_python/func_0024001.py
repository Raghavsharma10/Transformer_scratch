def objectatrib(instance, atrib):
    '''
    this filter is going to be useful to execute an object method or get an
    object attribute dynamically. this method is going to take into account
    the atrib param can contains underscores
    '''
    atrib = atrib.replace("__", ".")
    atribs = []
    atribs = atrib.split(".")

    obj = instance
    for atrib in atribs:
        if type(obj) == dict:
            result = obj[atrib]
        else:
            try:
                result = getattr(obj, atrib)()
            except Exception:
                result = getattr(obj, atrib)

        obj = result
    return result