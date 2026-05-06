def _find_files(dirpath: str) -> 'Iterable[str]':
    """Find files recursively.

    Returns a generator that yields paths in no particular order.
    """
    for dirpath, dirnames, filenames in os.walk(dirpath, topdown=True,
                                                followlinks=True):
        if os.path.basename(dirpath).startswith('.'):
            del dirnames[:]
        for filename in filenames:
            yield os.path.join(dirpath, filename)