def _read(self, command, future):
        """Invoked when a command is executed to read and parse its results.
        It will loop on the IOLoop until the response is complete and then
        set the value of the response in the execution future.

        :param command: The command that was being executed
        :type command: tredis.client.Command
        :param future: The execution future
        :type future: tornado.concurrent.Future

        """
        response = self._reader.gets()
        if response is not False:
            if isinstance(response, hiredis.ReplyError):
                if response.args[0].startswith('MOVED '):
                    self._on_cluster_data_moved(response.args[0], command,
                                                future)
                elif response.args[0].startswith('READONLY '):
                    self._on_read_only_error(command, future)
                else:
                    future.set_exception(exceptions.RedisError(response))
            elif command.callback is not None:
                future.set_result(command.callback(response))
            elif command.expectation is not None:
                self._eval_expectation(command, response, future)
            else:
                future.set_result(response)
        else:

            def on_data(data):
                # LOGGER.debug('Read %r', data)
                self._reader.feed(data)
                self._read(command, future)

            command.connection.read(on_data)