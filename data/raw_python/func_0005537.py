def url_mod(url: str, new_params: dict) -> str:
    """
    Modifies existing URL by setting/overriding specified query string parameters.
    Note: Does not support multiple querystring parameters with identical name.
    :param url: Base URL/path to modify
    :param new_params: Querystring parameters to set/override (dict)
    :return: New URL/path
    """
    from urllib.parse import urlparse, parse_qsl, urlunparse, urlencode
    res = urlparse(url)
    query_params = dict(parse_qsl(res.query))
    for k, v in new_params.items():
        if v is None:
            query_params[str(k)] = ''
        else:
            query_params[str(k)] = str(v)
    parts = list(res)
    parts[4] = urlencode(query_params)
    return urlunparse(parts)