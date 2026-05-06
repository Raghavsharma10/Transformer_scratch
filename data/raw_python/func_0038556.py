def overlay_url_for(endpoint, filename=None, **values):
    """
    Replace flasks url_for() function to allow usage without template changes

    If the requested endpoint is static or ending in .static, it tries to serve a bower asset, otherwise it will pass
    the arguments to flask.url_for()

    See http://flask.pocoo.org/docs/0.10/api/#flask.url_for
    """
    default_url_for_args = values.copy()
    if filename:
        default_url_for_args['filename'] = filename

    if endpoint == 'static' or endpoint.endswith('.static'):

        if os.path.sep in filename:
            filename_parts = filename.split(os.path.sep)
            component = filename_parts[0]
            # Using * magic here to expand list
            filename = os.path.join(*filename_parts[1:])

            returned_url = build_url(component, filename, **values)

            if returned_url is not None:
                return returned_url

    return None