def starts_expanded(name):
    """Return True if directory is a parent of initial cwd."""

    if name is '/':
        return True

    l = name.split(dir_sep())
    if len(l) > len(_initial_cwd):
        return False

    if l != _initial_cwd[:len(l)]:
        return False

    return True