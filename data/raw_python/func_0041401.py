def pfx_path(path):
    """ Prefix a path with the OS path separator if it is not already """
    if path[0] != os.path.sep: return os.path.sep + path
    else:                      return path