def getPathOfExecutable(executable):
    """
    Returns the full path of the executable, or None if the executable
    can not be found.
    """
    exe_paths = os.environ['PATH'].split(':')
    for exe_path in exe_paths:
        exe_file = os.path.join(exe_path, executable)
        if os.path.isfile(exe_file) and os.access(exe_file, os.X_OK):
            return exe_file
    return None