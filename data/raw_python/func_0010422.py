def check_network_connection(server, port):
    '''
    Checks if jasper can connect a network server.
    Arguments:
        server -- (optional) the server to connect with (Default:
                  "www.google.com")
    Returns:
        True or False
    '''
    logger = logging.getLogger(__name__)
    logger.debug("Checking network connection to server '%s'...", server)
    try:
        # see if we can resolve the host name -- tells us if there is
        # a DNS listening
        host = socket.gethostbyname(server)
        # connect to the host -- tells us if the host is actually
        # reachable
        sock = socket.create_connection((host, port), 2)
        sock.close()
    except Exception:  # pragma: no cover
        logger.debug("Network connection not working")
        return False
    logger.debug("Network connection working")
    return True