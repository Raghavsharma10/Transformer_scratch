def sign_url_path(url, secret_key, expire_in=None, digest=None):
    # type: (str, bytes, int, Callable) -> str
    """
    Sign a URL (excluding the domain and scheme).

    :param url: URL to sign
    :param secret_key: Secret key
    :param expire_in: Expiry time.
    :param digest: Specify the digest function to use; default is sha256 from hashlib
    :return: Signed URL

    """
    result = urlparse(url)
    query_args = MultiValueDict(parse_qs(result.query))
    query_args['_'] = token()
    if expire_in is not None:
        query_args['expires'] = int(time() + expire_in)
    query_args['signature'] = _generate_signature(result.path, secret_key, query_args, digest)
    return "%s?%s" % (result.path, urlencode(list(query_args.sorteditems(True))))