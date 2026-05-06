def connect(self, ip, port):
        """
        Connects to a server at the specified ip and port.

        :param ip: an IP address
        :type ip: str or unicode
        :param port: port number from 1024 up to 65535
        :type port: int
        :rtype: self
        """
        _check_valid_port_range(port)

        address = (ip, port)

        if address in self._addresses:
            error = 'Already connected to {0} on port {1}'.format(ip, port)
            log.exception(error)
            raise ValueError(error)

        self._addresses.append(address)

        if self._is_ready:
            _check_valid_num_connections(self._sock.socket_type,
                                         len(self._addresses))

            _connect_zmq_sock(self._sock, ip, port)

        return self