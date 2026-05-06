def cfgGetBool(theObj, name, dflt):
    """ Get a stringified val from a ConfigObj obj and return it as bool """
    strval = theObj.get(name, None)
    if strval is None:
        return dflt
    return strval.lower().strip() == 'true'