def find_command(cmd, path=None, pathext=None):
    """
    Taken `from Django http://bit.ly/1njB3Y9>`_.
    """
    if path is None:
        path = os.environ.get('PATH', '').split(os.pathsep)
    if isinstance(path, string_types):
        path = [path]

    # check if there are path extensions for Windows executables
    if pathext is None:
        pathext = os.environ.get('PATHEXT', '.COM;.EXE;.BAT;.CMD')
        pathext = pathext.split(os.pathsep)

    # don't use extensions if the command ends with one of them
    for ext in pathext:
        if cmd.endswith(ext):
            pathext = ['']
            break

    # check if we find the command on PATH
    for p in path:
        f = os.path.join(p, cmd)
        if os.path.isfile(f):
            return f
        for ext in pathext:
            fext = f + ext
            if os.path.isfile(fext):
                return fext
    return None