def _consume_queue(self):
        """
        Continually blocks until data is on the internal queue, then calls
        the session's registered callback and sends a PublishMessageReceived
        if callback returned True.
        """
        while True:
            session, block_id, raw_data = self._queue.get()
            data = json.loads(raw_data.decode('utf-8'))  # decode as JSON
            try:
                result = session.callback(data)
                if result is None:
                    self.log.warn("Callback %r returned None, expected boolean.  Messages "
                                  "are not marked as received unless True is returned", session.callback)
                elif result:
                    # Send a Successful PublishMessageReceived with the
                    # block id sent in request
                    if self._write_queue is not None:
                        response_message = struct.pack('!HHH',
                                                       PUBLISH_MESSAGE_RECEIVED,
                                                       block_id, 200)
                        self._write_queue.put((session.socket, response_message))
            except Exception as exception:
                self.log.exception(exception)

            self._queue.task_done()