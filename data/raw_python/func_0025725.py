def findAllCfgTasksUnderDir(aDir):
    """ Finds all installed tasks by examining any .cfg files found on disk
        at and under the given directory, as an installation might be.
        This returns a dict of { file name : task name }
    """
    retval = {}
    for f in irafutils.rglob(aDir, '*.cfg'):
        retval[f] = getEmbeddedKeyVal(f, TASK_NAME_KEY, '')
    return retval