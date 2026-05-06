def report_down(self, service, port):
        """
        Reports the given service's present node as down by deleting the
        node's znode in Zookeeper if the znode is present.

        Waits for the Zookeeper connection to be established before further
        action is taken.
        """
        wait_on_any(self.connected, self.shutdown)

        node = Node.current(service, port)

        path = self.path_of(service, node)
        try:
            logger.debug("Deleting znode at %s", path)
            self.client.delete(path)
        except exceptions.NoNodeError:
            pass