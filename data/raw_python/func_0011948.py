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
        :param args:
            The positional arguments to send along with the request. If the
            first argument is a generator, the request will be sent in chunks
            :ref:`(more info) <chunked-messages>`.
        :type args: tuple
        :param kwargs: keyword arguments to send along with the request
        :type kwargs: dict
        :param broadcast:
            if ``True``, send to every peer with a matching subscription
        :type broadcast: bool

        :returns:
            a :class:`RPC <junction.futures.RPC>` object representing the
            RPC and its future response.

        :raises:
            :class:`Unroutable <junction.errors.Unroutable>` if no peers are
            registered to receive the message
        '''
        rpc = self._dispatcher.send_rpc(service, routing_id, method,
                args or (), kwargs or {}, not broadcast)

        if not rpc:
            raise errors.Unroutable()

        return rpc