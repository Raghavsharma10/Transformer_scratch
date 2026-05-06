def validate_ip_address(ip_address):
    """Validate the ip_address

    :param ip_address: (str) IP address
    :return: (bool) True if the ip_address is valid
    """
    # Validate the IP address
    log = logging.getLogger(mod_logger + '.validate_ip_address')
    if not isinstance(ip_address, basestring):
        log.warn('ip_address argument is not a string')
        return False

    # Ensure there are 3 dots
    num_dots = 0
    for c in ip_address:
        if c == '.':
            num_dots += 1
    if num_dots != 3:
        log.info('Not a valid IP address: {i}'.format(i=ip_address))
        return False

    # Use the socket module to test
    try:
        socket.inet_aton(ip_address)
    except socket.error as e:
        log.info('Not a valid IP address: {i}\n{e}'.format(i=ip_address, e=e))
        return False
    else:
        log.info('Validated IP address: %s', ip_address)
        return True