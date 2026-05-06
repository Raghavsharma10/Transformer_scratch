def declare_queue(self, name='', *, durable=True, exclusive=False,
                      auto_delete=False, passive=False,
                      nowait=False, arguments=None):
        """
        Declare a queue on the broker. If the queue does not exist, it will be created.

        This method is a :ref:`coroutine <coroutine>`.

        :param str name: the name of the queue.
            Supplying a name of '' will create a queue with a unique name of the server's choosing.
        :keyword bool durable: If true, the queue will be re-created when the server restarts.
        :keyword bool exclusive: If true, the queue can only be accessed by the current connection,
            and will be deleted when the connection is closed.
        :keyword bool auto_delete: If true, the queue will be deleted when the last consumer is cancelled.
            If there were never any conusmers, the queue won't be deleted.
        :keyword bool passive: If true and queue with such a name does not
            exist it will raise a :class:`exceptions.NotFound` instead of
            creating it. Arguments ``durable``, ``auto_delete`` and
            ``exclusive`` are ignored if ``passive=True``.
        :keyword bool nowait: If true, will not wait for a declare-ok to arrive.
        :keyword dict arguments: Table of optional parameters for extensions to the AMQP protocol. See :ref:`extensions`.

        :return: The new :class:`Queue` object.
        """
        q = yield from self.queue_factory.declare(
            name, durable, exclusive, auto_delete, passive, nowait,
            arguments if arguments is not None else {})
        return q