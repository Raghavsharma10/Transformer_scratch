def handle_connection_change(self, state):
        """
        Callback for handling changes in the kazoo client's connection state.

        If the connection becomes lost or suspended, the `connected` Event
        is cleared.  Other given states imply that the connection is
        established so `connected` is set.
        """
        if state == client.KazooState.LOST:
            if not self.shutdown.is_set():
                logger.info("Zookeeper session lost!")
            self.connected.clear()
        elif state == client.KazooState.SUSPENDED:
            logger.info("Zookeeper connection suspended!")
            self.connected.clear()
        else:
            logger.info("Zookeeper connection (re)established.")
            self.connected.set()