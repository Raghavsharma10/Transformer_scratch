def execute(self, command, future):
        """Execute a command after connecting if necessary.

        :param bytes command: command to execute after the connection
            is established
        :param tornado.concurrent.Future future:  future to resolve
            when the command's response is received.

        """
        LOGGER.debug('execute(%r, %r)', command, future)
        if self.connected:
            self._write(command, future)
        else:

            def on_connected(cfuture):
                if cfuture.exception():
                    return future.set_exception(cfuture.exception())
                self._write(command, future)

            self.io_loop.add_future(self.connect(), on_connected)