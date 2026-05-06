def _send(self):
        """ Send a message lazy formatted with args.
        External log attributes can be passed via named attribute `extra`,
        like in logging from the standart library.

        Note:
            * Attrs must be dict, otherwise the whole message would be skipped.
            * The key field in an attr is converted to string.
            * The value is sent as is if isinstance of (str, unicode, int, float, long, bool),
              otherwise we convert the value to string.
        """
        buff = BytesIO()
        while True:
            msgs = list()
            try:
                msg = yield self.queue.get()

                # we need to connect first, as we issue verbosity request just after connection
                # and channels should strictly go in ascending order
                if not self._connected:
                    yield self.connect()

                try:
                    while True:
                        msgs.append(msg)
                        counter = next(self.counter)
                        msgpack_pack([counter, EMIT, msg], buff)
                        msg = self.queue.get_nowait()
                except queues.QueueEmpty:
                    pass

                try:
                    yield self.pipe.write(buff.getvalue())
                except Exception:
                    pass
                # clean the buffer or we will end up without memory
                buff.truncate(0)
            except Exception:
                for message in msgs:
                    self._log_to_fallback(message)