def _get_vispy_app_dir():
    """Helper to get the default directory for storing vispy data"""
    # Define default user directory
    user_dir = os.path.expanduser('~')

    # Get system app data dir
    path = None
    if sys.platform.startswith('win'):
        path1, path2 = os.getenv('LOCALAPPDATA'), os.getenv('APPDATA')
        path = path1 or path2
    elif sys.platform.startswith('darwin'):
        path = os.path.join(user_dir, 'Library', 'Application Support')
    # On Linux and as fallback
    if not (path and os.path.isdir(path)):
        path = user_dir

    # Maybe we should store things local to the executable (in case of a
    # portable distro or a frozen application that wants to be portable)
    prefix = sys.prefix
    if getattr(sys, 'frozen', None):  # See application_dir() function
        prefix = os.path.abspath(os.path.dirname(sys.path[0]))
    for reldir in ('settings', '../settings'):
        localpath = os.path.abspath(os.path.join(prefix, reldir))
        if os.path.isdir(localpath):
            try:
                open(os.path.join(localpath, 'test.write'), 'wb').close()
                os.remove(os.path.join(localpath, 'test.write'))
            except IOError:
                pass  # We cannot write in this directory
            else:
                path = localpath
                break

    # Get path specific for this app
    appname = '.vispy' if path == user_dir else 'vispy'
    path = os.path.join(path, appname)
    return path