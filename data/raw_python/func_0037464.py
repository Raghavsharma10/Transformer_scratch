def avatar_url_from_email(email, size=64, default='retro', dns=False):
    """
    Our own implementation since fas doesn't support this nicely yet.
    """

    if dns:
        # This makes an extra DNS SRV query, which can slow down our webapps.
        # It is necessary for libravatar federation, though.
        import libravatar
        return libravatar.libravatar_url(
            email=email,
            size=size,
            default=default,
        )
    else:
        params = _ordered_query_params([('s', size), ('d', default)])
        query = parse.urlencode(params)
        hash = md5(email.encode('utf-8')).hexdigest()
        return "https://seccdn.libravatar.org/avatar/%s?%s" % (hash, query)