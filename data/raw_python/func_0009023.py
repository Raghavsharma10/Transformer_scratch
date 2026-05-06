def _close_callback(self):
        """Callback called when redis closed the connection.

        The callback queue is emptied and we call each callback found
        with None or with an exception object to wake up blocked client.
        """
        while True:
            try:
                callback = self.__callback_queue.popleft()
                callback(ConnectionError("closed connection"))
            except IndexError:
                break
        if self.subscribed:
            # pubsub clients
            self._reply_list.append(ConnectionError("closed connection"))
            self._condition.notify_all()