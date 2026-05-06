def get_webpack(request, name='DEFAULT'):
    """
    Get the Webpack object for a given webpack config.

    Called at most once per request per config name.
    """
    if not hasattr(request, '_webpack_map'):
        request._webpack_map = {}
    wp = request._webpack_map.get(name)
    if wp is None:
        wp = request._webpack_map[name] = Webpack(request, name)
    return wp