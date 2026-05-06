def merge_dicts(a, b, path=None):
    """ Merge dict :b: into dict :a:

    Code snippet from http://stackoverflow.com/a/7205107
    """
    if path is None:
        path = []

    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                raise Exception(
                    'Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a