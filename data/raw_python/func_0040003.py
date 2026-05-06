def get_pathext(default_pathext=None):
    """Returns the path extensions from environment or a default"""
    if default_pathext is None:
        default_pathext = os.pathsep.join([ '.COM', '.EXE', '.BAT', '.CMD' ])
    pathext = os.environ.get('PATHEXT', default_pathext)
    return pathext