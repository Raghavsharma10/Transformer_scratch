def fullsplit(path, result=None, base_path=None):
    """
    Split a pathname into components (the opposite of os.path.join) in a
    platform-neutral way.
    """

    if base_path:
        path = path.replace(base_path, '')

    if result is None:
        result = []
    head, tail = os.path.split(path)
    if head == '':
        return [tail] + result
    if head == path:
        return result
    return fullsplit(head, [tail] + result)