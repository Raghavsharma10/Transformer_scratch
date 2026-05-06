def consume(self, callback, *, no_local=False, no_ack=False, exclusive=False, arguments=None):
        """
        Start a consumer on the queue. Messages will be delivered asynchronously to the consumer.
        The callback function will be called whenever a new message arrives on the queue.

        Advanced usage: the callback object must be callable
        (it must be a function or define a ``__call__`` method),
        but may also define some further methods:

        * ``callback.on_cancel()``: called with no parameters when the consumer is successfully cancelled.
        * ``callback.on_error(exc)``: called when the channel is closed due to an error.
          The argument passed is the exception which caused the error.

        This method is a :ref:`coroutine <coroutine>`.

        :param callable callback: a callback to be called when a message is delivered.
            The callback must accept a single argument (an instance of :class:`~asynqp.message.IncomingMessage`).
        :keyword bool no_local: If true, the server will not deliver messages that were
            published by this connection.
        :keyword bool no_ack: If true, messages delivered to the consumer don't require acknowledgement.
        :keyword bool exclusive: If true, only this consumer can access the queue.
        :keyword dict arguments: Table of optional parameters for extensions to the AMQP protocol. See :ref:`extensions`.

        :return: The newly created :class:`Consumer` object.
        """
        if self.deleted:
            raise Deleted("Queue {} was deleted".format(self.name))

        self.sender.send_BasicConsume(self.name, no_local, no_ack, exclusive, arguments or {})
        tag = yield from self.synchroniser.wait(spec.BasicConsumeOK)
        consumer = Consumer(
            tag, callback, self.sender, self.synchroniser, self.reader,
            loop=self._loop)
        self.consumers.add_consumer(consumer)
        self.reader.ready()
        return consumer