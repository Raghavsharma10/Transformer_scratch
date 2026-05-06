def cache_file(package, mode):
    """
    Yields a file-like object for the purpose of writing to or
    reading from the cache.

    The code:

        with cache_file(...) as f:
            # do stuff with f

    is guaranteed to convert any exceptions to warnings (*),
    both in the cache_file(...) call and the 'do stuff with f'
    block.

    The file is automatically closed upon exiting the with block.

    If getting an actual file fails, yields a DummyFile.

    :param package: the name of the package being checked as a string
    :param mode: the mode to open the file in, either 'r' or 'w'
    """

    f = DummyFile()

    # We have to wrap the whole function body in this block to guarantee
    # catching all exceptions. In particular the yield needs to be inside
    # to catch exceptions coming from the with block.
    with exception_to_warning('use cache while checking for outdated package',
                              OutdatedCacheFailedWarning):
        try:
            cache_path = os.path.join(tempfile.gettempdir(),
                                      get_cache_filename(package))
            if mode == 'w' or os.path.exists(cache_path):
                f = open(cache_path, mode)
        finally:
            # Putting the yield in the finally section ensures that exactly
            # one thing is yielded once, otherwise @contextmanager would
            # raise an exception.
            with f:  # closes the file afterards
                yield f