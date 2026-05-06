def connected_client(self):
        """Returns a ContextManagerFuture to be yielded in a with statement.

        Returns:
            A ContextManagerFuture object.

        Examples:
            >>> with (yield pool.connected_client()) as client:
                    # client is a connected tornadis.Client instance
                    # it will be automatically released to the pool thanks to
                    # the "with" keyword
                    reply = yield client.call("PING")
        """
        future = self.get_connected_client()
        cb = functools.partial(self._connected_client_release_cb, future)
        return ContextManagerFuture(future, cb)