def publish(self, service, routing_id, method, args=None, kwargs=None,
            broadcast=False, udp=False):
        '''Send a 1-way message

        :param service: the service name (the routing top level)
        :type service: anything hash-able
        :param int routing_id:
            the id used for routing within the registered handlers of the
            service
        :param string method: the method name to call
        :param tuple args:
            The positional arguments to send along with the request. If the
            first positional argument is a generator object, the publish will
            be sent in chunks :ref:`(more info) <chunked-messages>`.
        :param dict kwargs: keyword arguments to send along with the request
        :param bool broadcast:
            if ``True``, send to every peer with a matching subscription.
        :param bool udp: deliver the message over UDP instead of the usual TCP

        :returns: None. use 'rpc' methods for requests with responses.

        :raises:
            :class:`Unroutable <junction.errors.Unroutable>` if no peers are
            registered to receive the message
        '''
        if udp:
            func = self._dispatcher.send_publish_udp
        else:
            func = self._dispatcher.send_publish
        if not func(None, service, routing_id, method,
                args or (), kwargs or {}, singular=not broadcast):
            raise errors.Unroutable()