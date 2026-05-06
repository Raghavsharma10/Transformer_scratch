def _on_cluster_data_moved(self, response, command, future):
        """Process the ``MOVED`` response from a Redis cluster node.

        :param bytes response: The response from the Redis server
        :param command: The command that was being executed
        :type command: tredis.client.Command
        :param future: The execution future
        :type future: tornado.concurrent.Future

        """
        LOGGER.debug('on_cluster_data_moved(%r, %r, %r)', response, command,
                     future)
        parts = response.split(' ')
        name = '{}:{}'.format(*common.split_connection_host_port(parts[2]))
        LOGGER.debug('Moved to %r', name)
        if name not in self._cluster:
            raise exceptions.ConnectionError(
                '{} is not connected'.format(name))
        self._cluster[name].execute(
            command._replace(connection=self._cluster[name]), future)