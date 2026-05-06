def declare_exchange(self, name, type, *, durable=True, auto_delete=False,
                         passive=False, internal=False, nowait=False,
                         arguments=None):
        """
        Declare an :class:`Exchange` on the broker. If the exchange does not exist, it will be created.

        This method is a :ref:`coroutine <coroutine>`.

        :param str name: the name of the exchange.
        :param str type: the type of the exchange
            (usually one of ``'fanout'``, ``'direct'``, ``'topic'``, or ``'headers'``)
        :keyword bool durable: If true, the exchange will be re-created when
            the server restarts.
        :keyword bool auto_delete: If true, the exchange will be
            deleted when the last queue is un-bound from it.
        :keyword bool passive: If `true` and exchange with such a name does
            not exist it will raise a :class:`exceptions.NotFound`. If `false`
            server will create it. Arguments ``durable``, ``auto_delete`` and
            ``internal`` are ignored if `passive=True`.
        :keyword bool internal: If true, the exchange cannot be published to
            directly; it can only be bound to other exchanges.
        :keyword bool nowait: If true, the method will not wait for declare-ok
            to arrive and return right away.
        :keyword dict arguments: Table of optional parameters for extensions to
            the AMQP protocol. See :ref:`extensions`.

        :return: the new :class:`Exchange` object.
        """
        if name == '':
            return exchange.Exchange(self.reader, self.synchroniser, self.sender, name, 'direct', True, False, False)

        if not VALID_EXCHANGE_NAME_RE.match(name):
            raise ValueError(
                "Invalid exchange name.\n"
                "Valid names consist of letters, digits, hyphen, underscore, "
                "period, or colon, and do not begin with 'amq.'")

        self.sender.send_ExchangeDeclare(
            name, type, passive, durable, auto_delete, internal, nowait,
            arguments or {})
        if not nowait:
            yield from self.synchroniser.wait(spec.ExchangeDeclareOK)
            self.reader.ready()
        ex = exchange.Exchange(
            self.reader, self.synchroniser, self.sender, name, type, durable,
            auto_delete, internal)
        return ex