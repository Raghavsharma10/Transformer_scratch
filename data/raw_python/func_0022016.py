def _parse(reactor, directory, pemdir, *args, **kwargs):
    """
    Parse a txacme endpoint description.

    :param reactor: The Twisted reactor.
    :param directory: ``twisted.python.url.URL`` for the ACME directory to use
        for issuing certs.
    :param str pemdir: The path to the certificate directory to use.
    """
    def colon_join(items):
        return ':'.join([item.replace(':', '\\:') for item in items])
    sub = colon_join(list(args) + ['='.join(item) for item in kwargs.items()])
    pem_path = FilePath(pemdir).asTextMode()
    acme_key = load_or_create_client_key(pem_path)
    return AutoTLSEndpoint(
        reactor=reactor,
        directory=directory,
        client_creator=partial(Client.from_url, key=acme_key, alg=RS256),
        cert_store=DirectoryStore(pem_path),
        cert_mapping=HostDirectoryMap(pem_path),
        sub_endpoint=serverFromString(reactor, sub))