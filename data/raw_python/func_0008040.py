def get_normalized_request_string(method, url, nonce, params, ext='', body_hash=None):
    """
    Returns a normalized request string as described iN OAuth2 MAC spec.

    http://tools.ietf.org/html/draft-ietf-oauth-v2-http-mac-00#section-3.3.1
    """
    urlparts = urlparse.urlparse(url)
    if urlparts.query:
        norm_url = '%s?%s' % (urlparts.path, urlparts.query)
    elif params:
        norm_url = '%s?%s' % (urlparts.path, get_normalized_params(params))
    else:
        norm_url = urlparts.path

    if not body_hash:
        body_hash = get_body_hash(params)

    port = urlparts.port
    if not port:
        assert urlparts.scheme in ('http', 'https')

        if urlparts.scheme == 'http':
            port = 80
        elif urlparts.scheme == 'https':
            port = 443

    output = [nonce, method.upper(), norm_url, urlparts.hostname, port, body_hash, ext, '']

    return '\n'.join(map(str, output))