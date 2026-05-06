def url_to_ips(url, port=None, ipv6=False, connect_type=socket.SOCK_STREAM,
               proto=socket.IPPROTO_TCP, flags=0):
    """
    Provide a list of IP addresses, uses `socket.getaddrinfo`

    .. code:: python

        reusables.url_to_ips("example.com", ipv6=True)
        # ['2606:2800:220:1:248:1893:25c8:1946']

    :param url: hostname to resolve to IP addresses
    :param port: port to send to getaddrinfo
    :param ipv6: Return IPv6 address if True, otherwise IPv4
    :param connect_type: defaults to STREAM connection, can be 0 for all
    :param proto: defaults to TCP, can be 0 for all
    :param flags: additional flags to pass
    :return: list of resolved IPs
    """
    try:
        results = socket.getaddrinfo(url, port,
                                     (socket.AF_INET if not ipv6
                                      else socket.AF_INET6),
                                     connect_type,
                                     proto,
                                     flags)
    except socket.gaierror:
        logger.exception("Could not resolve hostname")
        return []

    return list(set([result[-1][0] for result in results]))