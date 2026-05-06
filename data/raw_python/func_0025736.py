def findTheLost(config_file, configspec_file, skipHidden=True):
    """ Find any lost/missing parameters in this cfg file, compared to what
    the .cfgspc says should be there. This method is recommended by the
    ConfigObj docs. Return a stringified list of item errors. """
    # do some sanity checking, but don't (yet) make this a serious error
    if not os.path.exists(config_file):
        print("ERROR: Config file not found: "+config_file)
        return []
    if not os.path.exists(configspec_file):
        print("ERROR: Configspec file not found: "+configspec_file)
        return []
    tmpObj = configobj.ConfigObj(config_file, configspec=configspec_file)
    simval = configobj.SimpleVal()
    test = tmpObj.validate(simval)
    if test == True:
        return []
    # If we get here, there is a dict returned of {key1: bool, key2: bool}
    # which matches the shape of the config obj.  We need to walk it to
    # find the Falses, since they are the missing pars.
    missing = []
    flattened = configobj.flatten_errors(tmpObj, test)
    # But, before we move on, skip/eliminate any 'hidden' items from our list,
    # since hidden items are really supposed to be missing from the .cfg file.
    if len(flattened) > 0 and skipHidden:
        keepers = []
        for tup in flattened:
            keep = True
            # hidden section
            if len(tup[0])>0 and isHiddenName(tup[0][-1]):
                keep = False
            # hidden par (in a section, or at the top level)
            elif tup[1] is not None and isHiddenName(tup[1]):
                keep = False
            if keep:
                keepers.append(tup)
        flattened = keepers
    flatStr = flattened2str(flattened, missing=True)
    return flatStr