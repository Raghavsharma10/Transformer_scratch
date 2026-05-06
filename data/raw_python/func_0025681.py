def osfn(filename):
    """Convert IRAF virtual path name to OS pathname."""

    # Try to emulate the CL version closely:
    #
    # - expands IRAF virtual file names
    # - strips blanks around path components
    # - if no slashes or relative paths, return relative pathname
    # - otherwise return absolute pathname
    if filename is None:
        return filename

    ename = Expand(filename)
    dlist = [part.strip() for part in ename.split(os.sep)]
    if len(dlist) == 1 and dlist[0] not in [os.curdir, os.pardir]:
        return dlist[0]

    # I use str.join instead of os.path.join here because
    # os.path.join("","") returns "" instead of "/"

    epath = os.sep.join(dlist)
    fname = os.path.abspath(epath)
    # append '/' if relative directory was at end or filename ends with '/'
    if fname[-1] != os.sep and dlist[-1] in ['', os.curdir, os.pardir]:
        fname = fname + os.sep
    return fname