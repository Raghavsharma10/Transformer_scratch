def rmtree_errorhandler(func, path, exc_info):
    """On Windows, the files in .svn are read-only, so when rmtree() tries to
    remove them, an exception is thrown.  We catch that here, remove the
    read-only attribute, and hopefully continue without problems."""
    exctype, value = exc_info[:2]
    # On Python 2.4, it will be OSError number 13
    # On all more recent Pythons, it'll be WindowsError number 5
    if not ((exctype is WindowsError and value.args[0] == 5) or
            (exctype is OSError and value.args[0] == 13)):
        raise
    # file type should currently be read only
    if ((os.stat(path).st_mode & stat.S_IREAD) != stat.S_IREAD):
        raise
    # convert to read/write
    os.chmod(path, stat.S_IWRITE)
    # use the original function to repeat the operation
    func(path)