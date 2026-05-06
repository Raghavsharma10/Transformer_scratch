def set_connections_params(
            self, harakiri=None, timeout_socket=None, retry_delay=None, timeout_headers=None, timeout_backend=None):
        """Sets connection-related parameters.

        :param int harakiri: Set gateway harakiri timeout (seconds).

        :param int timeout_socket: Node socket timeout (seconds).
            Used to set the SPDY timeout. This is the maximum amount of inactivity after
            the SPDY connection is closed.

            Default: 60.

        :param int retry_delay: Retry connections to dead static nodes after the specified
            amount of seconds. Default: 30.

        :param int timeout_headers: Defines the timeout (seconds) while waiting for http headers.

            Default: `socket_timeout`.

        :param int timeout_backend: Defines the timeout (seconds) when connecting to backend instances.

            Default: `socket_timeout`.

        """

        super(RouterHttp, self).set_connections_params(
            **filter_locals(locals(), ['timeout_headers', 'timeout_backend']))

        self._set_aliased('headers-timeout', timeout_headers)
        self._set_aliased('connect-timeout', timeout_backend)

        return self