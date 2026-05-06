def get_obj_attr_tuple(path):
    """Split path into (obj, attribute) tuple.

    Given `path` is 'os.path.exists' will thus return `(os.path, 'exists')`

    If path is not a str, delegates to `get_function_host(path)`

    """
    if not isinstance(path, str):
        return get_function_host(path)

    if path.startswith('.'):
        raise TypeError('relative imports are not supported')

    try:
        leading, end = path.rsplit('.', 1)
    except ValueError:
        raise TypeError('path must have dots')

    return get_obj(leading), end