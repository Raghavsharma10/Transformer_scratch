def unbind(self, arguments=None):
        """
        Unbind the queue from the exchange.

        This method is a :ref:`coroutine <coroutine>`.
        """
        if self.deleted:
            raise Deleted("Queue {} was already unbound from exchange {}".format(self.queue.name, self.exchange.name))

        self.sender.send_QueueUnbind(self.queue.name, self.exchange.name, self.routing_key, arguments or {})
        yield from self.synchroniser.wait(spec.QueueUnbindOK)
        self.deleted = True
        self.reader.ready()