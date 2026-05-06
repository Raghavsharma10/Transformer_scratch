def setPar(theDict, name, value):
    """ Sets a par's value without having to give its scope/section. """
    section, previousVal = findFirstPar(theDict, name)
    # "section" is the actual object, not a copy
    section[name] = value