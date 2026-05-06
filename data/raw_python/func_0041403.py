def file_or_default(path, default, function = None):
    """ Return a default value if a file does not exist """
    try:
        result = file_get_contents(path)
        if function != None: return function(result)
        return result
    except IOError as e:
        if e.errno == errno.ENOENT: return default
        raise