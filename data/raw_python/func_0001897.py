def disconnect(self, ip, port):
        """
        Disconnects from a server at the specified ip and port.

        :param ip: an IP address
        :type ip: str or unicode
        :param port: port number from 1024 up to 65535
        :type port: int
        :rtype: self
        """
        _check_valid_port_range(port)
        address = (ip, port)

        try:
            self._addresses.remove(address)
        except ValueError:
            error = 'There was no connection to {0} on port {1}'.format(ip, port)
            log.exception(error)
            raise ValueError(error)

        if self._is_ready:
            _disconnect_zmq_sock(self._sock, ip, port)

        return self