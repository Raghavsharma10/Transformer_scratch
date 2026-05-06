def get_ip_info(ip: str, exceptions: bool=False, timeout: int=10) -> tuple:
    """
    Returns (ip, country_code, host) tuple of the IP address.
    :param ip: IP address
    :param exceptions: Raise Exception or not
    :param timeout: Timeout in seconds. Note that timeout only affects geo IP part, not getting host name.
    :return: (ip, country_code, host)
    """
    import traceback
    import socket
    if not ip:  # localhost
        return None, '', ''
    host = ''
    country_code = get_geo_ip(ip, exceptions=exceptions, timeout=timeout).get('country_code', '')
    try:
        res = socket.gethostbyaddr(ip)
        host = res[0][:255] if ip else ''
    except Exception as e:
        msg = 'socket.gethostbyaddr({}) failed: {}'.format(ip, traceback.format_exc())
        logger.error(msg)
        if exceptions:
            raise e
    return ip, country_code, host