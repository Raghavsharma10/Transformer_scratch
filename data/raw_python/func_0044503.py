def url_is_from_any_domain(url, domains):
    """Return True if the url belongs to any of the given domains"""
    host = parse_url(url).netloc.lower()

    if host:
        return any(((host == d.lower()) or (host.endswith('.%s' % d.lower())) for d in domains))
    else:
        return False