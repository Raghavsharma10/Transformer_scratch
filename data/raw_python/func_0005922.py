def find_project_dir():
    """Runs up the stack to find the location of manage.py
    which will be considered a project base path.

    :rtype: str|unicode
    """
    frame = inspect.currentframe()

    while True:
        frame = frame.f_back
        fname = frame.f_globals['__file__']

        if os.path.basename(fname) == 'manage.py':
            break

    return os.path.dirname(fname)