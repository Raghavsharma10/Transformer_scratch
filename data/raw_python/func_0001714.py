def api_post(message, url, name, http_data=None, auth=None):
    """ Send `message` as `name` to `url`.
    You can specify extra variables in `http_data`
    """
    try:
        import requests
    except ImportError:
        print('requests is required to do api post.', file=sys.stderr)
        sys.exit(1)

    data = {name : message}
    if http_data:
        for var in http_data:
            key, value = var.split('=')
            data[key] = value

    response = requests.post(
        url,
        data=data,
        auth=auth
    )

    if response.status_code != 200:
        raise RuntimeError('Unable to post data')