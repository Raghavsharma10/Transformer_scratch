def getCfgFilesInDirForTask(aDir, aTask, recurse=False):
    """ This is a specialized function which is meant only to keep the
        same code from needlessly being much repeated throughout this
        application.  This must be kept as fast and as light as possible.
        This checks a given directory for .cfg files matching a given
        task.  If recurse is True, it will check subdirectories.
        If aTask is None, it returns all files and ignores aTask.
    """
    if recurse:
        flist = irafutils.rglob(aDir, '*.cfg')
    else:
        flist = glob.glob(aDir+os.sep+'*.cfg')
    if aTask:
        retval = []
        for f in flist:
            try:
                if aTask == getEmbeddedKeyVal(f, TASK_NAME_KEY, ''):
                    retval.append(f)
            except Exception as e:
                print('Warning: '+str(e))
        return retval
    else:
        return flist