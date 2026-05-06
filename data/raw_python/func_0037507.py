def _choose_read_fs(authority, cache, read_path, version_check, hasher):
    '''
    Context manager returning the appropriate up-to-date readable filesystem

    Use ``cache`` if it is a valid filessystem and has a file at
    ``read_path``, otherwise use ``authority``. If the file at
    ``read_path`` is out of date, update the file in ``cache`` before
    returning it.
    '''

    if cache and cache.fs.isfile(read_path):
        if version_check(hasher(cache.fs.open(read_path, 'rb'))):
            yield cache.fs

        elif authority.fs.isfile(read_path):
            fs.utils.copyfile(
                authority.fs,
                read_path,
                cache.fs,
                read_path)
            yield cache.fs

        else:
            _makedirs(authority.fs, fs.path.dirname(read_path))
            _makedirs(cache.fs, fs.path.dirname(read_path))
            yield cache.fs

    else:
        if not authority.fs.isfile(read_path):
            _makedirs(authority.fs, fs.path.dirname(read_path))

        yield authority.fs