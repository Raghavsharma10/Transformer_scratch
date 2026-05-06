def glob(*args):
    """Returns list of paths matching one or more wildcard patterns.

    Args:
      include_dirs: Include directories in the output
    """
    if len(args) is 1 and isinstance(args[0], list):
        args = args[0]
    matches = []
    for pattern in args:
        for item in glob2.glob(pattern):
            if not os.path.isdir(item):
                matches.append(item)
    return matches