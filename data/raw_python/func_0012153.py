def send_rpc(self, service, routing_id, method, args=None, kwargs=None,
            broadcast=False):
        '''Send out an RPC request

        :param service: the service name (the routing top level)
        :type service: anything hash-able
        :param routing_id:
            The id used for routing within the registered handlers of the
            service.
        :type routing_id: int
        :param method: the method name to call
        :type method: string
        :param args: the positional arguments to send along with the request
        :type args: tuple
        :param kwargs: keyword arguments to send along with the request
        :type kwargs: dict
        :param broadcast:
            if ``True``, send to all peers with matching subscriptions
        :type broadcast: bool

        :returns:
            a :class:`RPC <junction.futures.RPC>` object representing the
            RPC and its future response.

        :raises:
            :class:`Unroutable <junction.errors.Unroutable>` if the client
            doesn't have a connection to a hub
        '''
        if not self._peer.up:
            raise errors.Unroutable()

        return self._dispatcher.send_proxied_rpc(service, routing_id, method,
                args or (), kwargs or {}, not broadcast)