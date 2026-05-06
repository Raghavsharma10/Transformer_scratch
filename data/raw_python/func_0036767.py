def current(cls, service, port):
        """
        Returns a Node instance representing the current service node.

        Collects the host and IP information for the current machine and
        the port information from the given service.
        """
        host = socket.getfqdn()
        return cls(
            host=host,
            ip=socket.gethostbyname(host),
            port=port,
            metadata=service.metadata
        )