def asHandle(fileNameOrHandle, mode='r'):
    """
    Decorator for file opening that makes it easy to open compressed files.
    Based on L{Bio.File.as_handle}.

    @param fileNameOrHandle: Either a C{str} or a file handle.
    @return: A generator that can be turned into a context manager via
        L{contextlib.contextmanager}.
    """
    if isinstance(fileNameOrHandle, six.string_types):
        if fileNameOrHandle.endswith('.gz'):
            if six.PY3:
                yield gzip.open(fileNameOrHandle, mode='rt', encoding='UTF-8')
            else:
                yield gzip.GzipFile(fileNameOrHandle)
        elif fileNameOrHandle.endswith('.bz2'):
            if six.PY3:
                yield bz2.open(fileNameOrHandle, mode='rt', encoding='UTF-8')
            else:
                yield bz2.BZ2File(fileNameOrHandle)
        else:
            with open(fileNameOrHandle) as fp:
                yield fp
    else:
        yield fileNameOrHandle