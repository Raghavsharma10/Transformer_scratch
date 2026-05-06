def set_basic_params(
            self, workers=None, zerg_server=None, fallback_node=None, concurrent_events=None,
            cheap_mode=None, stats_server=None, quiet=None, buffer_size=None,
            fallback_nokey=None, subscription_key=None, emperor_command_socket=None):
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

        :param bool fallback_nokey: Move to fallback node even if a subscription key is not found.

        :param str|unicode subscription_key: Skip uwsgi parsing and directly set a key.

        :param str|unicode emperor_command_socket: Set the emperor command socket that will receive spawn commands.

            See `.empire.set_emperor_command_params()`.

        """
        super(RouterFast, self).set_basic_params(**filter_locals(locals(), [
            'fallback_nokey',
            'subscription_key',
            'emperor_command_socket',
        ]))

        self._set_aliased('fallback-on-no-key', fallback_nokey, cast=bool)
        self._set_aliased('force-key', subscription_key)
        self._set_aliased('emperor-socket', emperor_command_socket)

        return self