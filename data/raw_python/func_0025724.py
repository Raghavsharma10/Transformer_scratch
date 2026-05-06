def findCfgFileForPkg(pkgName, theExt, pkgObj=None, taskName=None):
    """ Locate the configuration files for/from/within a given python package.
    pkgName is a string python package name.  This is used unless pkgObj
    is given, in which case pkgName is taken from pkgObj.__name__.
    theExt is either '.cfg' or '.cfgspc'. If the task name is known, it is
    given as taskName, otherwise one is determined using the pkgName.
    Returns a tuple of (package-object, cfg-file-name). """
    # arg check
    ext = theExt
    if ext[0] != '.': ext = '.'+theExt

    # Do the import, if needed
    pkgsToTry = {}
    if pkgObj:
        pkgsToTry[pkgObj.__name__] = pkgObj
    else:
        # First try something simple like a regular or dotted import
        try:
            fl = []
            if pkgName.find('.') > 0:
                fl = [ pkgName[:pkgName.rfind('.')], ]
            pkgsToTry[str(pkgName)] = __import__(str(pkgName), fromlist=fl)
        except:
            throwIt = True
            # One last case to try is something like "csc_kill" from
            # "acstools.csc_kill", but this convenience capability will only be
            # allowed if the parent pkg (acstools) has already been imported.
            if isinstance(pkgName, string_types) and pkgName.find('.') < 0:
                matches = [x for x in sys.modules.keys() \
                           if x.endswith("."+pkgName)]
                if len(matches)>0:
                    throwIt = False
                    for mmm in matches:
                        pkgsToTry[mmm] = sys.modules[mmm]
            if throwIt:
                raise NoCfgFileError("Unfound package or "+ext+" file via: "+\
                                     "import "+str(pkgName))

    # Now that we have the package object (or a few of them to try), for each
    # one find the .cfg or .cfgspc file, and return
    # Return as soon as ANY match is found.
    for aPkgName in pkgsToTry:
        aPkg = pkgsToTry[aPkgName]
        path = os.path.dirname(aPkg.__file__)
        if len(path) < 1: path = '.'
        flist = irafutils.rglob(path, "*"+ext)
        if len(flist) < 1:
            continue

        # Go through these and find the first one for the assumed or given task
        # name.  The task name for 'BigBlackBox.drizzle' would be 'drizzle'.
        if taskName is None:
            taskName = aPkgName.split(".")[-1]
        flist.sort()
        for f in flist:
            # A .cfg file gets checked for _task_name_=val, but a .cfgspc file
            # will have a string check function signature as the val.
            if ext == '.cfg':
                itsTask = getEmbeddedKeyVal(f, TASK_NAME_KEY, '')
            else: # .cfgspc
                sigStr  = getEmbeddedKeyVal(f, TASK_NAME_KEY, '')
                # .cfgspc file MUST have an entry for TASK_NAME_KEY w/ a default
                itsTask = vtor_checks.sigStrToKwArgsDict(sigStr)['default']
            if itsTask == taskName:
                # We've found the correct file in an installation area.  Return
                # the package object and the found file.
                return aPkg, f

    # What, are you still here?
    raise NoCfgFileError('No valid '+ext+' files found in package: "'+ \
                         str(pkgName)+'" for task: "'+str(taskName)+'"')