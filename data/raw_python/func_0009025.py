def call(self, *args, **kwargs):
        """Calls a redis command and returns a Future of the reply.

        Args:
            *args: full redis command as variable length argument list or
                a Pipeline object (as a single argument).
            **kwargs: internal private options (do not use).

        Returns:
            a Future with the decoded redis reply as result (when available) or
                a ConnectionError object in case of connection error.

        Raises:
            ClientError: your Pipeline object is empty.

        Examples:

            >>> @tornado.gen.coroutine
                def foobar():
                    client = Client()
                    result = yield client.call("HSET", "key", "field", "val")
        """
        if not self.is_connected():
            if self.autoconnect:
                # We use this method only when we are not contected
                # to void performance penaly due to gen.coroutine decorator
                return self._call_with_autoconnect(*args, **kwargs)
            else:
                error = ConnectionError("you are not connected and "
                                        "autoconnect=False")
                return tornado.gen.maybe_future(error)
        return self._call(*args, **kwargs)