def _create_cluster_connection(self, node):
        """Create a connection to a Redis server.

        :param node: The node to connect to
        :type node: tredis.cluster.ClusterNode

        """
        LOGGER.debug('Creating a cluster connection to %s:%s', node.ip,
                     node.port)
        conn = _Connection(
            node.ip,
            node.port,
            0,
            self._read,
            self._on_closed,
            self.io_loop,
            cluster_node=True,
            read_only='slave' in node.flags,
            slots=node.slots)
        self.io_loop.add_future(conn.connect(), self._on_connected)