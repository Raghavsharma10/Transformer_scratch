def delete(self, *, if_unused=True):
        """
        Delete the exchange.

        This method is a :ref:`coroutine <coroutine>`.

        :keyword bool if_unused: If true, the exchange will only be deleted if
            it has no queues bound to it.
        """
        self.sender.send_ExchangeDelete(self.name, if_unused)
        yield from self.synchroniser.wait(spec.ExchangeDeleteOK)
        self.reader.ready()