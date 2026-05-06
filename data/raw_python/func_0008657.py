def purge(self):
        """
        Purge all undelivered messages from the queue.

        This method is a :ref:`coroutine <coroutine>`.
        """
        self.sender.send_QueuePurge(self.name)
        yield from self.synchroniser.wait(spec.QueuePurgeOK)
        self.reader.ready()