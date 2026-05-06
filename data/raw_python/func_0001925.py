def safe_makedirs(fdir):
    """
    Make an arbitrary directory.  This is safe to call for Python 2 users.

    :param fdir: Directory path to make.
    :return:
    """
    if os.path.isdir(fdir):
        pass
        # print 'dir already exists: %s' % str(dir)
    else:
        try:
            os.makedirs(fdir)
        except WindowsError as e:
            if 'Cannot create a file when that file already exists' in e:
                log.debug('relevant dir already exists')
            else:
                raise WindowsError(e)
    return True