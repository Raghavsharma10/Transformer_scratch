def get(self, *, no_ack=False):
        """
        Synchronously get a message from the queue.

        This method is a :ref:`coroutine <coroutine>`.

        :keyword bool no_ack: if true, the broker does not require acknowledgement of receipt of the message.

        :return: an :class:`~asynqp.message.IncomingMessage`,
            or ``None`` if there were no messages on the queue.
        """
        if self.deleted:
            raise Deleted("Queue {} was deleted".format(self.name))

        self.sender.send_BasicGet(self.name, no_ack)
        tag_msg = yield from self.synchroniser.wait(spec.BasicGetOK, spec.BasicGetEmpty)

        if tag_msg is not None:
            consumer_tag, msg = tag_msg
            assert consumer_tag is None
        else:
            msg = None
        self.reader.ready()
        return msg