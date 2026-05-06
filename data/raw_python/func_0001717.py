def content_from_path(path, encoding='utf-8'):
    """Return the content of the specified file as a string.

    This function also supports loading resources from packages.
    """
    if not os.path.isabs(path) and ':' in path:
        package, path = path.split(':', 1)
        content = resource_string(package, path)
    else:
        path = os.path.expanduser(path)
        with open(path, 'rb') as fh:
            content = fh.read()

    return content.decode(encoding)