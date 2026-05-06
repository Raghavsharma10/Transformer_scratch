def _read_callback(self, data=None):
        """Callback called when some data are read on the socket.

        The buffer is given to the hiredis parser. If a reply is complete,
        we put the decoded reply to on the reply queue.

        Args:
            data (str): string (buffer) read on the socket.
        """
        try:
            if data is not None:
                self.__reader.feed(data)
                while True:
                    reply = self.__reader.gets()
                    if reply is not False:
                        try:
                            callback = self.__callback_queue.popleft()
                            # normal client (1 reply = 1 callback)
                            callback(reply)
                        except IndexError:
                            # pubsub clients
                            self._reply_list.append(reply)
                            self._condition.notify_all()
                    else:
                        break
        except hiredis.ProtocolError:
            # something nasty occured (corrupt stream => no way to recover)
            LOG.warning("corrupted stream => disconnect")
            self.disconnect()