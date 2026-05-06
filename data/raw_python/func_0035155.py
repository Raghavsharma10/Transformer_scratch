def find_executable(name):
    """
    Finds the actual path to a named command.

    The first one on $PATH wins.
    """
    for pt in os.environ.get('PATH', '').split(':'):
        candidate = os.path.join(pt, name)
        if os.path.exists(candidate):
            return candidate