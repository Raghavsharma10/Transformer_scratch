def set_connections_params(self, harakiri=None, timeout_socket=None, retry_delay=None, retry_max=None):
        """Sets connection-related parameters.

        :param int harakiri: Set gateway harakiri timeout (seconds).

        :param int timeout_socket: Node socket timeout (seconds). Default: 60.

        :param int retry_delay: Retry connections to dead static nodes after the specified
            amount of seconds. Default: 30.

        :param int retry_max: Maximum number of retries/fallbacks to other nodes. Default: 3.

        """
        super(RouterSsl, self).set_connections_params(**filter_locals(locals(), ['retry_max']))

        self._set_aliased('max-retries', retry_max)

        return self