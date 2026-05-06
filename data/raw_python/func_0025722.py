def getObjectFromTaskArg(theTask, strict, setAllToDefaults):
    """ Take the arg (usually called theTask), which can be either a subclass
    of ConfigObjPars, or a string package name, or a .cfg filename - no matter
    what it is - take it and return a ConfigObjPars object.
    strict - bool - warning severity, passed to the ConfigObjPars() ctor
    setAllToDefaults - bool - if theTask is a pkg name, force all to defaults
    """

    # Already in the form we need (instance of us or of subclass)
    if isinstance(theTask, ConfigObjPars):
        if setAllToDefaults:
            raise RuntimeError('Called getObjectFromTaskArg with existing'+\
                  ' object AND setAllToDefaults - is unexpected use case.')
        # If it is an existing object, make sure it's internal param list is
        # up to date with it's ConfigObj dict, since the user may have manually
        # edited the dict before calling us.
        theTask.syncParamList(False) # use strict somehow?
        # Note - some validation is done here in IrafPar creation, but it is
        # not the same validation done by the ConfigObj s/w (no check funcs).
        # Do we want to do that too here?
        return theTask

    # For example, a .cfg file
    if os.path.isfile(str(theTask)):
        try:
            return ConfigObjPars(theTask, strict=strict,
                                 setAllToDefaults=setAllToDefaults)
        except KeyError:
            # this might just be caused by a file sitting in the local cwd with
            # the same exact name as the package we want to import, let's see
            if theTask.find('.') > 0: # it has an extension, like '.cfg'
                raise # this really was an error
            # else we drop down to the next step - try it as a pkg name

    # Else it must be a Python package name to load
    if isinstance(theTask, str) and setAllToDefaults:
        # NOTE how we pass the task name string in setAllToDefaults
        return ConfigObjPars('', setAllToDefaults=theTask, strict=strict)
    else:
        return getParsObjForPyPkg(theTask, strict)