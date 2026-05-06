def check_extracted_paths(namelist, subdir=None):
    """
    Check whether zip file paths are all relative, and optionally in a
    specified subdirectory, raises an exception if not

    namelist: A list of paths from the zip file
    subdir: If specified then check whether all paths in the zip file are
      under this subdirectory

    Python docs are unclear about the security of extract/extractall:
    https://docs.python.org/2/library/zipfile.html#zipfile.ZipFile.extractall
    https://docs.python.org/2/library/zipfile.html#zipfile.ZipFile.extract
    """
    def relpath(p):
        # relpath strips a trailing sep
        # Windows paths may also use unix sep
        q = os.path.relpath(p)
        if p.endswith(os.path.sep) or p.endswith('/'):
            q += os.path.sep
        return q

    parent = os.path.abspath('.')
    if subdir:
        if os.path.isabs(subdir):
            raise FileException('subdir must be a relative path', subdir)
        subdir = relpath(subdir + os.path.sep)

    for name in namelist:
        if os.path.commonprefix([parent, os.path.abspath(name)]) != parent:
            raise FileException('Insecure path in zipfile', name)

        if subdir and os.path.commonprefix(
                [subdir, relpath(name)]) != subdir:
            raise FileException(
                'Path in zipfile is not in required subdir', name)