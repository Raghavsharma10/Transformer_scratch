def create_binding(self, vhost, exchange, queue, rt_key=None, args=None):
        """
        Creates a binding between an exchange and a queue on a given vhost.

        :param string vhost: vhost housing the exchange/queue to bind
        :param string exchange: the target exchange of the binding
        :param string queue: the queue to bind to the exchange
        :param string rt_key: the routing key to use for the binding
        :param list args: extra arguments to associate w/ the binding.
        :returns: boolean
        """

        vhost = quote(vhost, '')
        exchange = quote(exchange, '')
        queue = quote(queue, '')
        body = json.dumps({'routing_key': rt_key, 'arguments': args or []})
        path = Client.urls['bindings_between_exch_queue'] % (vhost,
                                                             exchange,
                                                             queue)
        binding = self._call(path, 'POST', body=body,
                                    headers=Client.json_headers)
        return binding