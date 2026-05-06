def _endpoint_to_target(self, endpoint):
        """Convert a URL into a host / port, or into a path to a unix domain socket

        Args:
            endpoint (str): A URL parsable by urlparse

        Returns:
            3 item tuple: (host, port, path).
            host and port will None, and path will be not None if a a unix domain socket URL is passed
            path will be None if a normal TCP based URL is passed

        """
        parsed = urlparse.urlparse(endpoint)
        scheme = parsed[0]
        hostport = parsed[1]

        if 'unix' in scheme:
            return (None, None, unquote(hostport))

        if scheme == 'https':
            target_port = 443
        else:
            target_port = 80

        (target_host, target_port) = self._split_hostport(hostport, default_port=target_port)
        return (target_host, target_port, None)