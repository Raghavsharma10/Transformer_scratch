def metadata_urls(identifiers, protocol=None, hosts=None):
    """An Archive.org metadata URL generator.

    :param identifiers: A set of Archive.org identifiers for which to
                        make metadata URLs.
    :type identifiers: iterable

    :param protocol: (optional) The HTTP protocol to use. "https://" is
                     used by default.
    :type protocol: str

    :param hosts: (optional) A set of hosts. A host will be chosen at
                  random. The default host is "archive.org".
    :type hosts: iterable

    :returns: A generator yielding Archive.org metadata URLs.
    """
    for identifier in identifiers:
        path = '/metadata/{}'.format(identifier)
        url = make_url(path, protocol, hosts)
        yield url