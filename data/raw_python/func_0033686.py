def run(self):
        """
        A context manager starting up threads to send and receive data from a
        gateway and handle callbacks. Yields when a connection has been made,
        and cleans up connections and threads when it's done.
        """
        listener_thr = _spawn(self.receiver.run)
        callback_thr = _spawn(self.callbacks.run)
        sender_thr = _spawn(self.sender.run)
        logger_thr = _spawn(self.logger.run)

        self.connect()
        try:
            yield
        finally:
            self.stop()

            # Wait for the listener to finish.
            listener_thr.join()
            self.callbacks.put('shutdown')

            # Tell the other threads to finish, and wait for them.
            for obj in [self.callbacks, self.sender, self.logger]:
                obj.stop()
            for thr in [callback_thr, sender_thr, logger_thr]:
                thr.join()