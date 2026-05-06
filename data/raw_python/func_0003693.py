def matchFileOnDirPath(curpath, pathdir):
    """Find match for a file by slicing away its directory elements
       from the front and replacing them with pathdir.  Assume that the
       end of curpath is right and but that the beginning may contain
       some garbage (or it may be short)
       Overlaps are allowed:
       e.g /tmp/fdjsklf/real/path/elements, /all/the/real/ =>
       /all/the/real/path/elements (assuming that this combined
       path exists)
    """
    if os.path.exists(curpath):
        return curpath
    filedirs = curpath.split('/')[1:]
    filename = filedirs[-1]
    filedirs = filedirs[:-1]
    if pathdir[-1] == '/':
        pathdir = pathdir[:-1]
    # assume absolute paths
    pathdirs = pathdir.split('/')[1:]
    lp = len(pathdirs)
    # Cut off matching file elements from the ends of the two paths
    for x in range(1, min(len(filedirs), lp)):
        # XXX this will not work if you have
        # /usr/foo/foo/filename.py
        if filedirs[-1] == pathdirs[-x]:
            filedirs = filedirs[:-1]
        else:
            break

    # Now cut try cuting off incorrect initial elements of curpath
    while filedirs:
        tmppath = '/' + '/'.join(pathdirs + filedirs + [filename])
        if os.path.exists(tmppath):
            return tmppath
        filedirs = filedirs[1:]
    tmppath = '/' + '/'.join(pathdirs + [filename])
    if os.path.exists(tmppath):
        return tmppath
    return None