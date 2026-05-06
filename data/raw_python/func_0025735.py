def mergeConfigObj(configObj, inputDict):
    """ Merge the inputDict values into an existing given configObj instance.
    The inputDict is a "flat" dict - it has no sections/sub-sections.  The
    configObj may have sub-sections nested to any depth.  This will raise a
    DuplicateKeyError if one of the inputDict keys is used more than once in
    configObj (e.g. within two different sub-sections). """
    # Expanded upon Warren's version in astrodrizzle

    # Verify that all inputDict keys in configObj are unique within configObj
    for key in inputDict:
        if countKey(configObj, key) > 1:
            raise DuplicateKeyError(key)
    # Now update configObj with each inputDict item
    for key in inputDict:
        setPar(configObj, key, inputDict[key])