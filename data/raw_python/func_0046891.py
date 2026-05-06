def _split_hostport(self, hostport, default_port=None):
        """Split a string in the format of '<host>:<port>' into it's component parts

        default_port will be used if a port is not included in the string

        Args:
            str ('<host>' or '<host>:<port>'): A string to split into it's parts

        Returns:
            two item tuple: (host, port)

        Raises:
            ValueError: The string was in an invalid element
        """

        try:
            (host, port) = hostport.split(':', 1)
        except ValueError:  # no colon in the string so make our own port
            host = hostport

            if default_port is None:
                raise ValueError('No port found in hostport, and default_port not provided.')

            port = default_port

        try:
            port = int(port)
            if port < 1 or port > 65535:
                raise ValueError()
        except ValueError:
            raise ValueError("{0} is not a valid TCP port".format(port))

        return (host, port)