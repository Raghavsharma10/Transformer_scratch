def connect(self):
        """
        Connect to one of the Disque nodes.

        You can get current connection with connected_node property

        :returns: nothing
        """
        self.connected_node = None

        for i, node in self.nodes.items():
            host, port = i.split(':')
            port = int(port)
            redis_client = redis.Redis(host, port, **self.client_kw_args)

            try:
                ret = redis_client.execute_command('HELLO')
                format_version, node_id = ret[0], ret[1]
                others = ret[2:]
                self.nodes[i] = Node(node_id, host, port, redis_client)
                self.connected_node = self.nodes[i]
            except redis.exceptions.ConnectionError:
                pass

        if not self.connected_node:
            raise ConnectionError('couldnt connect to any nodes')
        logger.info("connected to node %s" % self.connected_node)