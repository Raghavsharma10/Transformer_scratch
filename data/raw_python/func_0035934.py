def TemporaryDirectory(suffix=None, prefix=None, dir=None, on_error='ignore'):  # @ReservedAssignment
    '''
    An extension to `tempfile.TemporaryDirectory`.

    Unlike with `python:tempfile`, a :py:class:`~pathlib.Path` is yielded on
    ``__enter__``, not a `str`.

    Parameters
    ----------
    suffix : str
        See `tempfile.TemporaryDirectory`.
    prefix : str
        See `tempfile.TemporaryDirectory`.
    dir : ~pathlib.Path
        See `tempfile.TemporaryDirectory`, but pass a :py:class:`~pathlib.Path` instead.
    on_error : str
        Handling of failure to delete directory (happens frequently on NFS), one of:

        raise
            Raise exception on failure.
        ignore
            Fail silently.
    '''
    if dir:
        dir = str(dir)  # @ReservedAssignment
    temp_dir = tempfile.TemporaryDirectory(suffix, prefix, dir)
    try:
        yield Path(temp_dir.name)
    finally:
        try:
            temp_dir.cleanup()
        except OSError as ex:
            print(ex)
            # Suppress relevant errors if ignoring failed delete
            if on_error != 'ignore' or ex.errno != errno.ENOTEMPTY:
                raise