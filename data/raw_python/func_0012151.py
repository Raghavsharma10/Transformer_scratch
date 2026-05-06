def publish(self, service, routing_id, method, args=None, kwargs=None,
            broadcast=False):
        '''Send a 1-way message

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
            if ``True``, send to every peer with a matching subscription
        :type broadcast: bool

        :returns: None. use 'rpc' methods for requests with responses.

        :raises:
            :class:`Unroutable <junction.errors.Unroutable>` if the client
            doesn't have a connection to a hub
        '''
        if not self._peer.up:
            raise errors.Unroutable()

        self._dispatcher.send_proxied_publish(service, routing_id, method,
                args or (), kwargs or {}, singular=not broadcast)