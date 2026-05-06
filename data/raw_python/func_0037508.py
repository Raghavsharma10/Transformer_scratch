def _get_write_fs():
    '''
    Context manager returning a writable filesystem

    Use a temporary directory and clean on exit.

    .. todo::

        Evaluate options for using a cached memoryFS or streaming object
        instead of an OSFS(tmp). This could offer significant performance
        improvements. Writing to the cache is less of a problem since this
        would be done in any case, though performance could be improved by
        writing to an in-memory filesystem and then writing to both cache and
        auth.

    '''

    tmp = tempfile.mkdtemp()

    try:
        # Create a writeFS and path to the directory containing the archive
        write_fs = OSFS(tmp)

        try:

            yield write_fs

        finally:
            _close(write_fs)

    finally:
        shutil.rmtree(tmp)