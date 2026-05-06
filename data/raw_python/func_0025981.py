def setWritePrivs(fname, makeWritable, ignoreErrors=False):
    """ Set a file named fname to be writable (or not) by user, with the
    option to ignore errors.  There is nothing ground-breaking here, but I
    was annoyed with having to repeate this little bit of code. """
    privs = os.stat(fname).st_mode
    try:
        if makeWritable:
            os.chmod(fname, privs | stat.S_IWUSR)
        else:
            os.chmod(fname, privs & (~ stat.S_IWUSR))
    except OSError:
        if ignoreErrors:
            pass # just try, don't whine
        else:
            raise