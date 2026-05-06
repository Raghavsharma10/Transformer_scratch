def set_connections_params(self, harakiri=None, timeout_socket=None):
        """Sets connection-related parameters.

        :param int harakiri: Set gateway harakiri timeout (seconds).

        :param int timeout_socket: Node socket timeout (seconds). Default: 60.

        """
        self._set_aliased('harakiri', harakiri)
        self._set_aliased('timeout', timeout_socket)

        return self