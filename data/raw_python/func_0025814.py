def diffFromDefaults(theTask, report=False):
    """ Load the given file (or existing object), and return a dict
    of its values which are different from the default values.  If report
    is set, print to stdout the differences. """
    # get the 2 dicts (trees: dicts of dicts)
    defaultTree = load(theTask, canExecute=False, strict=True, defaults=True)
    thisTree    = load(theTask, canExecute=False, strict=True, defaults=False)
    # they must be flattenable
    defaultFlat = cfgpars.flattenDictTree(defaultTree)
    thisFlat    = cfgpars.flattenDictTree(thisTree)
    # use the "set" operations till there is a dict.diff()
    # thanks to:  http://stackoverflow.com/questions/715234
    diffFlat = dict( set(thisFlat.items()) - \
                     set(defaultFlat.items()) )
    if report:
        defaults_of_diffs_only = {}
#       { k:defaultFlat[k] for k in diffFlat.keys() }
        for k in diffFlat:
            defaults_of_diffs_only[k] = defaultFlat[k]
        msg = 'Non-default values of "'+str(theTask)+'":\n'+ \
              _flat2str(diffFlat)+ \
              '\n\nDefault values:\n'+ \
              _flat2str(defaults_of_diffs_only)
        print(msg)
    return diffFlat