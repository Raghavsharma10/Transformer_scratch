def wrap_file(file_like_obj):
    """Wrap a file like object in an async stream wrapper.

    Files generated with `open()` may be one of several types. This
    convenience function retruns the stream wrapped in the most appropriate
    wrapper for the type. If the stream is already wrapped it is returned
    unaltered.
    """
    if isinstance(file_like_obj, AsyncIOBaseWrapper):

        return file_like_obj

    if isinstance(file_like_obj, sync_io.FileIO):

        return AsyncFileIOWrapper(file_like_obj)

    if isinstance(file_like_obj, sync_io.BufferedRandom):

        return AsyncBufferedRandomWrapper(file_like_obj)

    if isinstance(file_like_obj, sync_io.BufferedReader):

        return AsyncBufferedReaderWrapper(file_like_obj)

    if isinstance(file_like_obj, sync_io.BufferedWriter):

        return AsyncBufferedWriterWrapper(file_like_obj)

    if isinstance(file_like_obj, sync_io.TextIOWrapper):

        return AsyncTextIOWrapperWrapper(file_like_obj)

    raise TypeError(
        'Unrecognized file stream type {}.'.format(file_like_obj.__class__),
    )