def bind(self, exchange, routing_key, *, arguments=None):
        """
        Bind a queue to an exchange, with the supplied routing key.

        This action 'subscribes' the queue to the routing key; the precise meaning of this
        varies with the exchange type.

        This method is a :ref:`coroutine <coroutine>`.

        :param asynqp.Exchange exchange: the :class:`Exchange` to bind to
        :param str routing_key: the routing key under which to bind
        :keyword dict arguments: Table of optional parameters for extensions to the AMQP protocol. See :ref:`extensions`.

        :return: The new :class:`QueueBinding` object
        """
        if self.deleted:
            raise Deleted("Queue {} was deleted".format(self.name))
        if not exchange:
            raise InvalidExchangeName("Can't bind queue {} to the default exchange".format(self.name))

        self.sender.send_QueueBind(self.name, exchange.name, routing_key, arguments or {})
        yield from self.synchroniser.wait(spec.QueueBindOK)
        b = QueueBinding(self.reader, self.sender, self.synchroniser, self, exchange, routing_key)
        self.reader.ready()
        return b