def set_basic_params(
            self, workers=None, zerg_server=None, fallback_node=None, concurrent_events=None,
            cheap_mode=None, stats_server=None, quiet=None, buffer_size=None,
            keepalive=None, resubscribe_addresses=None):
        """
        :param int workers: Number of worker processes to spawn.

        :param str|unicode zerg_server: Attach the router to a zerg server.

        :param str|unicode fallback_node: Fallback to the specified node in case of error.

        :param int concurrent_events: Set the maximum number of concurrent events router can manage.

            Default: system dependent.

        :param bool cheap_mode: Enables cheap mode. When the router is in cheap mode,
            it will not respond to requests until a node is available.
            This means that when there are no nodes subscribed, only your local app (if any) will respond.
            When all of the nodes go down, the router will return in cheap mode.

        :param str|unicode stats_server: Router stats server address to run at.

        :param bool quiet: Do not report failed connections to instances.

        :param int buffer_size: Set internal buffer size in bytes. Default: page size.

        :param int keepalive: Allows holding the connection open even if the request has a body.

            * http://uwsgi.readthedocs.io/en/latest/HTTP.html#http-keep-alive

            .. note:: See http11 socket type for an alternative.

        :param str|unicode|list[str|unicode] resubscribe_addresses: Forward subscriptions
            to the specified subscription server.


        """
        super(RouterHttp, self).set_basic_params(**filter_locals(locals(), drop=[
            'keepalive',
            'resubscribe_addresses',
        ]))

        self._set_aliased('keepalive', keepalive)
        self._set_aliased('resubscribe', resubscribe_addresses, multi=True)

        return self