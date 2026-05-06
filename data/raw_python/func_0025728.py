def getUsrCfgFilesForPyPkg(pkgName):
    """ See if the user has one of their own local .cfg files for this task,
        such as might be created automatically during the save of a read-only
        package, and return their names. """
    # Get the python package and it's .cfg file
    thePkg, theFile = findCfgFileForPkg(pkgName, '.cfg')
    # See if the user has any of their own local .cfg files for this task
    tname = getEmbeddedKeyVal(theFile, TASK_NAME_KEY)
    flist = getCfgFilesInDirForTask(getAppDir(), tname)
    return flist