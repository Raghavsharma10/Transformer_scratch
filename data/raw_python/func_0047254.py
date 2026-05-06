def _generate_signature(url_path, secret_key, query_args, digest=None, encoder=None):
    # type: (str, bytes, Dict[str, str], Callable, Callable) -> str
    """
    Generate signature from pre-parsed URL.
    """
    digest = digest or DEFAULT_DIGEST
    encoder = encoder or DEFAULT_ENCODER
    msg = "%s?%s" % (url_path, '&'.join('%s=%s' % i for i in query_args.sorteditems(multi=True)))
    if _compat.text_type:
        msg = msg.encode('UTF8')
    signature = hmac.new(secret_key, msg, digestmod=digest).digest()
    if _compat.PY2:
        return encoder(signature).rstrip('=')  # Strip padding
    else:
        return encoder(signature).decode().rstrip('=')