def search(path, matcher="*", dirs=False, files=True):
    """Recursive search function.

    Args:
        path (str): Path to search recursively
        matcher (str or callable): String pattern to search for or function
            that returns True/False for a file argument
        dirs (bool): if True returns directories that match the pattern
        files(bool): if True returns files that match the patter

    Yields:
        str: Found files and directories
    """
    if callable(matcher):

        def fnmatcher(items):
            return list(filter(matcher, items))

    else:

        def fnmatcher(items):
            return fnmatch.filter(items, matcher)

    for root, directories, filenames in os.walk(os.path.abspath(path)):
        to_match = []
        if dirs:
            to_match.extend(directories)
        if files:
            to_match.extend(filenames)
        for item in fnmatcher(to_match):
            yield os.path.join(root, item)