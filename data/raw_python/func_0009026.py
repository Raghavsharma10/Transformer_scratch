def async_call(self, *args, **kwargs):
        """Calls a redis command, waits for the reply and call a callback.

        Following options are available (not part of the redis command itself):

        - callback
            Function called (with the result as argument) when the result
            is available. If not set, the reply is silently discarded. In
            case of errors, the callback is called with a
            TornadisException object as argument.

        Args:
            *args: full redis command as variable length argument list or
                a Pipeline object (as a single argument).
            **kwargs: options as keyword parameters.

        Examples:

            >>> def cb(result):
                    pass
            >>> client.async_call("HSET", "key", "field", "val", callback=cb)
        """
        def after_autoconnect_callback(future):
            if self.is_connected():
                self._call(*args, **kwargs)
            else:
                # FIXME
                pass

        if 'callback' not in kwargs:
            kwargs['callback'] = discard_reply_cb
        if not self.is_connected():
            if self.autoconnect:
                connect_future = self.connect()
                cb = after_autoconnect_callback
                self.__connection._ioloop.add_future(connect_future, cb)
            else:
                error = ConnectionError("you are not connected and "
                                        "autoconnect=False")
                kwargs['callback'](error)
        else:
            self._call(*args, **kwargs)