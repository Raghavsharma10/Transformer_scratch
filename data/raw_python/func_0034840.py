def _on_cluster_discovery(self, future):
        """Invoked when the Redis server has responded to the ``CLUSTER_NODES``
        command.

        :param future: The future containing the response from Redis
        :type future: tornado.concurrent.Future

        """
        LOGGER.debug('_on_cluster_discovery(%r)', future)
        common.maybe_raise_exception(future)
        nodes = future.result()
        for node in nodes:
            name = '{}:{}'.format(node.ip, node.port)
            if name in self._cluster:
                LOGGER.debug('Updating cluster connection info for %s:%s',
                             node.ip, node.port)
                self._cluster[name].set_slots(node.slots)
                self._cluster[name].set_read_only('slave' in node.flags)
            else:
                self._create_cluster_connection(node)
        self._discovery = True