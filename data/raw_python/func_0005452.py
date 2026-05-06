def match_globs(path, patterns):
    # type: (str, List[str]) -> bool
    """ Test whether the given *path* matches any patterns in *patterns*

    Args:
        path (str):
            A file path to test for matches.
        patterns (list[str]):
            A list of glob string patterns to test against. If *path* matches
            any of those patters, it will return True.

    Returns:
        bool: **True** if the *path* matches any pattern in *patterns*.
    """
    for pattern in (p for p in patterns if p):
        if pattern.startswith('/'):
            regex = fnmatch.translate(pattern[1:])

            temp_path = path[1:] if path.startswith('/') else path

            m = re.search(regex, temp_path)

            if m and m.start() == 0:
                return True

        elif fnmatch.fnmatch(path, pattern):
                return True

    return False