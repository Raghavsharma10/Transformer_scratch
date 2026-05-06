def find_subdirs(startdir='.', recursion_depth=None):
    """Find all subdirectory of a directory.

    Inputs:
        startdir: directory to start with. Defaults to the current folder.
        recursion_depth: number of levels to traverse. None is infinite.

    Output: a list of absolute names of subfolders.

    Examples:
        >>> find_subdirs('dir',0)  # returns just ['dir']

        >>> find_subdirs('dir',1)  # returns all direct (first-level) subdirs
                                   # of 'dir'.
    """
    startdir = os.path.expanduser(startdir)
    direct_subdirs = [os.path.join(startdir, x) for x in os.listdir(
        startdir) if os.path.isdir(os.path.join(startdir, x))]
    if recursion_depth is None:
        next_recursion_depth = None
    else:
        next_recursion_depth = recursion_depth - 1
    if (recursion_depth is not None) and (recursion_depth <= 1):
        return [startdir] + direct_subdirs
    else:
        subdirs = []
        for d in direct_subdirs:
            subdirs.extend(find_subdirs(d, next_recursion_depth))
        return [startdir] + subdirs