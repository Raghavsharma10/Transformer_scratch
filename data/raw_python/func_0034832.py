def _write(self, command, future):
        """Write a command to the socket

        :param Command command: the Command data structure

        """

        def on_written():
            self._on_written(command, future)

        try:
            self._stream.write(command.command, callback=on_written)
        except iostream.StreamClosedError as error:
            future.set_exception(exceptions.ConnectionError(error))
        except Exception as error:
            LOGGER.exception('unhandled write failure - %r', error)
            future.set_exception(exceptions.ConnectionError(error))