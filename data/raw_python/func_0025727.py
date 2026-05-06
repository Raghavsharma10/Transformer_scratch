def getParsObjForPyPkg(pkgName, strict):
    """ Locate the appropriate ConfigObjPars (or subclass) within the given
        package. NOTE this begins the same way as getUsrCfgFilesForPyPkg().
        Look for .cfg file matches in these places, in this order:
          1 - any named .cfg file in current directory matching given task
          2 - if there exists a ~/.teal/<taskname>.cfg file
          3 - any named .cfg file in SOME*ENV*VAR directory matching given task
          4 - the installed default .cfg file (with the given package)
    """
    # Get the python package and it's .cfg file - need this no matter what
    installedPkg, installedFile = findCfgFileForPkg(pkgName, '.cfg')
    theFile = None
    tname = getEmbeddedKeyVal(installedFile, TASK_NAME_KEY)

    # See if the user has any of their own .cfg files in the cwd for this task
    if theFile is None:
        flist = getCfgFilesInDirForTask(os.getcwd(), tname)
        if len(flist) > 0:
            if len(flist) == 1: # can skip file times sort
                theFile = flist[0]
            else:
                # There are a few different choices.  In the absence of
                # requirements to the contrary, just take the latest.  Set up a
                # list of tuples of (mtime, fname) so we can sort by mtime.
                ftups = [ (os.stat(f)[stat.ST_MTIME], f) for f in flist]
                ftups.sort()
                theFile = ftups[-1][1]

    # See if the user has any of their own app-dir .cfg files for this task
    if theFile is None:
        flist = getCfgFilesInDirForTask(getAppDir(), tname) # verifies tname
        flist = [f for f in flist if os.path.basename(f) == tname+'.cfg']
        if len(flist) > 0:
            theFile = flist[0]
            assert len(flist) == 1, str(flist) # should never happen

    # Add code to check an env. var defined area?  (speak to users first)

    # Did we find one yet?  If not, use the installed version
    useInstVer = False
    if theFile is None:
        theFile = installedFile
        useInstVer = True

    # Create a stand-in instance from this file.  Force a read-only situation
    # if we are dealing with the installed, (expected to be) unwritable file.
    return ConfigObjPars(theFile, associatedPkg=installedPkg,
                         forceReadOnly=useInstVer, strict=strict)