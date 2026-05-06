def read_file(rel_path, paths=None, raw=False, as_list=False, as_iter=False,
              *args, **kwargs):
    '''
        find a file that lives somewhere within a set of paths and
        return its contents. Default paths include 'static_dir'
    '''
    if not rel_path:
        raise ValueError("rel_path can not be null!")
    paths = str2list(paths)
    # try looking the file up in a directory called static relative
    # to SRC_DIR, eg assuming metrique git repo is in ~/metrique
    # we'd look in ~/metrique/static
    paths.extend([STATIC_DIR, os.path.join(SRC_DIR, 'static')])
    paths = [os.path.expanduser(p) for p in set(paths)]
    for path in paths:
        path = os.path.join(path, rel_path)
        logger.debug("trying to read: %s " % path)
        if os.path.exists(path):
            break
    else:
        raise IOError("path %s does not exist!" % rel_path)
    args = args if args else ['rU']
    fd = open(path, *args, **kwargs)
    if raw:
        return fd

    if as_iter:
        return read_in_chunks(fd)
    else:
        fd_lines = fd.readlines()

    if as_list:
        return fd_lines
    else:
        return ''.join(fd_lines)