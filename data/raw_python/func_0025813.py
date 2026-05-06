def unlearn(taskPkgName, deleteAll=False):
    """ Find the task named taskPkgName, and delete any/all user-owned
    .cfg files in the user's resource directory which apply to that task.
    Like a unix utility, this returns 0 on success (no files found or only
    1 found but deleted).  For multiple files found, this uses deleteAll,
    returning the file-name-list if deleteAll is False (to indicate the
    problem) and without deleting any files.  MUST check return value.
    This does not prompt the user or print to the screen. """

    # this WILL throw an exception if the taskPkgName isn't found
    flist = cfgpars.getUsrCfgFilesForPyPkg(taskPkgName) # can raise
    if flist is None or len(flist) == 0:
        return 0
    if len(flist) == 1:
        os.remove(flist[0])
        return 0
    # at this point, we know more than one matching file was found
    if deleteAll:
        for f in flist:
            os.remove(f)
        return 0
    else:
        return flist