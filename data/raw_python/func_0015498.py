def split_host_port(cls, server):
        """
        Return (host, port) from server.

        Port defaults to 11211.

        >>> split_host_port('127.0.0.1:11211')
        ('127.0.0.1', 11211)
        >>> split_host_port('127.0.0.1')
        ('127.0.0.1', 11211)
        """
        host, port = splitport(server)
        if port is None:
            port = 11211
        port = int(port)
        if re.search(':.*$', host):
            host = re.sub(':.*$', '', host)
        return host, port