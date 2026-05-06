def set_resubscription_params(self, addresses=None, bind_to=None):
        """You can specify a dgram address (udp or unix) on which all of the subscriptions
        request will be forwarded to (obviously changing the node address to the router one).

        The system could be useful to build 'federated' setup.

        * http://uwsgi.readthedocs.io/en/latest/Changelog-2.0.1.html#resubscriptions

        :param str|unicode|list[str|unicode] addresses: Forward subscriptions to the specified subscription server.

        :param str|unicode|list[str|unicode] bind_to: Bind to the specified address when re-subscribing.

        """
        self._set_aliased('resubscribe', addresses, multi=True)
        self._set_aliased('resubscribe-bind', bind_to)

        return self