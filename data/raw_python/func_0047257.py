def verify_url(url, secret_key, **kwargs):
    """
    Verify a signed URL (excluding the domain and scheme).

    :param url: URL to sign
    :param secret_key: Secret key
    :rtype: bool
    :raises: URLError

    """
    result = urlparse(url)
    query_args = MultiValueDict(parse_qs(result.query))
    return verify_url_path(result.path, query_args, secret_key, **kwargs)