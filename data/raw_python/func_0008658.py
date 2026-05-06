def delete(self, *, if_unused=True, if_empty=True):
        """
        Delete the queue.

        This method is a :ref:`coroutine <coroutine>`.

        :keyword bool if_unused: If true, the queue will only be deleted
            if it has no consumers.
        :keyword bool if_empty: If true, the queue will only be deleted if
            it has no unacknowledged messages.
        """
        if self.deleted:
            raise Deleted("Queue {} was already deleted".format(self.name))

        self.sender.send_QueueDelete(self.name, if_unused, if_empty)
        yield from self.synchroniser.wait(spec.QueueDeleteOK)
        self.deleted = True
        self.reader.ready()