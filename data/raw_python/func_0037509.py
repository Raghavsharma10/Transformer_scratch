def _prepare_write_fs(read_fs, cache, read_path, readwrite_mode=True):
    '''
    Prepare a temporary filesystem for writing to read_path

    The file will be moved to write_path on close if modified.
    '''

    with _get_write_fs() as write_fs:

        # If opening in read/write or append mode, make sure file data is
        # accessible
        if readwrite_mode:

            if not write_fs.isfile(read_path):
                _touch(write_fs, read_path)

                if read_fs.isfile(read_path):
                    fs.utils.copyfile(
                        read_fs, read_path, write_fs, read_path)

        else:
            _touch(write_fs, read_path)

        yield write_fs