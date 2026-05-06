def extract_domain(host):
    """
    Domain name extractor. Turns host names into domain names, ported
    from pwdhash javascript code"""
    host = re.sub('https?://', '', host)
    host = re.match('([^/]+)', host).groups()[0]
    domain = '.'.join(host.split('.')[-2:])
    if domain in _domains:
        domain = '.'.join(host.split('.')[-3:])
    return domain