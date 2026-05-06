def get_local_path(
        authority,
        cache,
        update,
        version_check,
        hasher,
        read_path,
        write_path=None,
        cache_on_write=False):
    '''
    Context manager for retrieving a system path for I/O and updating on change


    Parameters
    ----------
    authority : object

        :py:mod:`pyFilesystem` filesystem object to use as the authoritative,
        up-to-date source for the archive

    cache : object

        :py:mod:`pyFilesystem` filesystem object to use as the cache. Default
        ``None``.

    use_cache : bool

         update, service_path, version_check, \*\*kwargs
    '''

    if write_path is None:
        write_path = read_path

    with _choose_read_fs(
            authority, cache, read_path, version_check, hasher) as read_fs:

        with _prepare_write_fs(
                read_fs, cache, read_path, readwrite_mode=True) as write_fs:

            yield write_fs.getsyspath(read_path)

            if write_fs.isfile(read_path):

                info = write_fs.getinfokeys(read_path, 'size')
                if 'size' in info:
                    if info['size'] == 0:
                        return

                with write_fs.open(read_path, 'rb') as f:
                    checksum = hasher(f)

                if not version_check(checksum):

                    if (
                        cache_on_write or
                        (
                            cache
                            and (
                                fs.path.abspath(read_path) ==
                                fs.path.abspath(write_path))
                            and cache.fs.isfile(read_path)
                        )
                    ):

                        _makedirs(cache.fs, fs.path.dirname(write_path))
                        fs.utils.copyfile(
                            write_fs, read_path, cache.fs, write_path)

                        _makedirs(authority.fs, fs.path.dirname(write_path))
                        fs.utils.copyfile(
                            cache.fs, write_path, authority.fs, write_path)
                    else:
                        _makedirs(authority.fs, fs.path.dirname(write_path))
                        fs.utils.copyfile(
                            write_fs, read_path, authority.fs, write_path)
                    update(**checksum)

            else:
                raise OSError(
                    'Local file removed during execution. '
                    'Archive not updated.')