def ida_connect(host='localhost', port=18861, retry=10):
    """
    Connect to an instance of IDA running our server.py.

    :param host:        The host to connect to
    :param port:        The port to connect to
    :param retry:       How many times to try after errors before giving up
    """
    for i in range(retry):
        try:
            LOG.debug('Connectint to %s:%d, try %d...', host, port, i + 1)
            link = rpyc_classic.connect(host, port)
            link.eval('2 + 2')
        except socket.error:
            time.sleep(1)
            continue
        else:
            LOG.debug('Connected to %s:%d', host, port)
            return link

    raise IDALinkError("Could not connect to %s:%d after %d tries" % (host, port, retry))