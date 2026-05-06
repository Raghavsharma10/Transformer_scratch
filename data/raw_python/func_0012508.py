def ip_to_url(ip_addr):
    """
    Resolve a hostname based off an IP address.

    This is very limited and will
    probably not return any results if it is a shared IP address or an
    address with improperly setup DNS records.

    .. code:: python

        reusables.ip_to_url('93.184.216.34') # example.com
        # None

        reusables.ip_to_url('8.8.8.8')
        # 'google-public-dns-a.google.com'


    :param ip_addr: IP address to resolve to hostname
    :return: string of hostname or None
    """
    try:
        return socket.gethostbyaddr(ip_addr)[0]
    except (socket.gaierror, socket.herror):
        logger.exception("Could not resolve hostname")