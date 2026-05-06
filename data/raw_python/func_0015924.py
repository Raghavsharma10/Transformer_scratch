def delete_binding(self, vhost, exchange, queue, rt_key):
        """
        Deletes a binding between an exchange and a queue on a given vhost.

        :param string vhost: vhost housing the exchange/queue to bind
        :param string exchange: the target exchange of the binding
        :param string queue: the queue to bind to the exchange
        :param string rt_key: the routing key to use for the binding
        """

        vhost = quote(vhost, '')
        exchange = quote(exchange, '')
        queue = quote(queue, '')
        body = ''
        path = Client.urls['rt_bindings_between_exch_queue'] % (vhost,
                                                                exchange,
                                                                queue,
                                                                rt_key)
        return self._call(path, 'DELETE', headers=Client.json_headers)