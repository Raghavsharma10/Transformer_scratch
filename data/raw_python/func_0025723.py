def getEmbeddedKeyVal(cfgFileName, kwdName, dflt=None):
    """ Read a config file and pull out the value of a given keyword. """
    # Assume this is a ConfigObj file.  Use that s/w to quickly read it and
    # put it in dict format.  Assume kwd is at top level (not in a section).
    # The input may also be a .cfgspc file.
    #
    # Only use ConfigObj here as a tool to generate a dict from a file - do
    # not use the returned object as a ConfigObj per se.  As such, we can call
    # with "simple" format, ie. no cfgspc, no val'n, and "list_values"=False.
    try:
        junkObj = configobj.ConfigObj(cfgFileName, list_values=False)
    except:
        if kwdName == TASK_NAME_KEY:
            raise KeyError('Can not parse as a parameter config file: '+ \
                           '\n\t'+os.path.realpath(cfgFileName))
        else:
            raise KeyError('Unfound key "'+kwdName+'" while parsing: '+ \
                           '\n\t'+os.path.realpath(cfgFileName))

    if kwdName in junkObj:
        retval = junkObj[kwdName]
        del junkObj
        return retval
    # Not found
    if dflt is not None:
        del junkObj
        return dflt
    else:
        if kwdName == TASK_NAME_KEY:
            raise KeyError('Can not parse as a parameter config file: '+ \
                           '\n\t'+os.path.realpath(cfgFileName))
        else:
            raise KeyError('Unfound key "'+kwdName+'" while parsing: '+ \
                           '\n\t'+os.path.realpath(cfgFileName))